from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from nerve.compilation import (
    ModelCompileCancelled,
    ModelCompileError,
    read_json,
)
from nerve.representation_optimizer.contracts import (
    BENCHMARK_RECORD_SCHEMA,
    REPRESENTATION_CANDIDATE_SCHEMA,
    ContractDocument,
)
from nerve.representation_optimizer.lifecycle import (
    CandidateState,
    OptimizationSession,
)
from nerve.representation_optimizer.providers.types import (
    ProviderCandidatePlan,
)
from nerve.representation_optimizer.staging.integrity import (
    integrity_evidence,
    validate_staged_candidate,
)
from nerve.representation_optimizer.staging.loading import (
    load_staged_candidate,
)
from nerve.representation_optimizer.validation.contracts import (
    PREBENCHMARK_RECORD_SCHEMA,
    ValidationPlan,
    ValidationRun,
)
from nerve.representation_optimizer.validation.evaluation import (
    build_prebenchmark_record,
    build_validation_record,
)
from nerve.representation_optimizer.validation.proofs import (
    ProofVerifierRegistry,
)
from nerve.representation_optimizer.validation.product_performance import (
    qualify_whole_model_product_performance,
)
from nerve.representation_optimizer.validation.protocols import (
    BehavioralValidationAdapter,
)
from nerve.representation_optimizer.validation.runner import (
    execute_validation_stage,
)
from nerve.representation_optimizer.validation.storage import (
    publish_prebenchmark_evidence,
    publish_validation_evidence,
)


@dataclass(frozen=True)
class CandidatePrebenchmarkOutcome:
    status: str
    plan: ValidationPlan
    record: ContractDocument
    sanity_run: ValidationRun | None
    evidence_path: Path
    session: OptimizationSession


@dataclass(frozen=True)
class CandidateValidationOutcome:
    status: str
    plan: ValidationPlan
    record: ContractDocument
    runs: tuple[ValidationRun, ...]
    evidence_path: Path
    session: OptimizationSession


def prepare_candidate_for_benchmark(
    *,
    package_dir: Path,
    candidate_workspace_root: Path,
    validation_workspace_root: Path,
    candidate_plan: ProviderCandidatePlan,
    construction_record: ContractDocument,
    validation_plan: ValidationPlan,
    session: OptimizationSession,
    proof_verifiers: ProofVerifierRegistry,
    adapter: BehavioralValidationAdapter,
    cancel_requested: Callable[[], bool] | None = None,
) -> CandidatePrebenchmarkOutcome:
    _require_state(
        session,
        candidate_plan.candidate_id,
        CandidateState.STAGED,
        "candidate must be staged before prebenchmark validation",
    )
    _validate_plan_inputs(
        candidate_plan,
        construction_record,
        validation_plan,
    )
    static_validation = {
        "status": "failed",
        "staged_integrity_digest": None,
        "artifact_count": 0,
    }
    working_session = session
    proof_results = ()
    sanity_run = None
    failure_reason: str | None = None
    terminal_state = CandidateState.REJECTED
    try:
        static_validation = _validate_static_candidate(
            package_dir=package_dir,
            candidate_workspace_root=candidate_workspace_root,
            candidate_plan=candidate_plan,
            construction_record=construction_record,
        )
        working_session = working_session.transition_candidate(
            candidate_plan.candidate_id,
            CandidateState.STATICALLY_VALIDATED,
            evidence_refs=(
                (
                    "ready/"
                    f"{candidate_plan.candidate_id}/integrity.json"
                ),
            ),
            reason=(
                "candidate contracts, construction record, source seal, "
                "artifacts, and staged integrity all passed"
            ),
        )
        if validation_plan.proofs:
            proof_results = proof_verifiers.prove(validation_plan)
            if not proof_results or any(
                result.status != "proven" for result in proof_results
            ):
                failure_reason = (
                    "one or more exact algebraic proof obligations "
                    "were not proven"
                )
            else:
                sanity_run = execute_validation_stage(
                    validation_plan,
                    stage="sanity",
                    adapter=adapter,
                    cancel_requested=cancel_requested,
                )
        else:
            sanity_run = execute_validation_stage(
                validation_plan,
                stage="sanity",
                adapter=adapter,
                cancel_requested=cancel_requested,
            )
        if sanity_run is not None and sanity_run.status != "completed":
            failure_reason = (
                "candidate failed cheap numerical or state-transition sanity"
            )
            if sanity_run.status == "cancelled":
                terminal_state = CandidateState.CANCELLED
    except ModelCompileCancelled as error:
        failure_reason = str(error)
        terminal_state = CandidateState.CANCELLED
    except Exception as error:
        failure_reason = str(error)
        terminal_state = CandidateState.FAILED

    record = build_prebenchmark_record(
        plan=validation_plan,
        static_validation=static_validation,
        proof_results=proof_results,
        sanity_run=sanity_run,
        failure_reason=failure_reason,
    )
    evidence_path = publish_prebenchmark_evidence(
        validation_workspace_root,
        plan=validation_plan,
        record=record,
        sanity_run=sanity_run,
        artifact_source=adapter,
        proof_artifact_source=proof_verifiers,
    )
    evidence_ref = (
        f"prebenchmark/{record.to_json()['prebenchmark_id']}/record.json",
    )
    if record.to_json()["status"] == "passed":
        next_session = working_session.transition_candidate(
            candidate_plan.candidate_id,
            CandidateState.PREBENCHMARK_VALIDATED,
            evidence_refs=evidence_ref,
            reason=(
                "static integrity, proof obligations, and cheap behavioral "
                "sanity all passed"
            ),
        )
        status = "passed"
    else:
        current = _candidate_state(
            working_session,
            candidate_plan.candidate_id,
        )
        allowed_terminal = (
            terminal_state
            if terminal_state == CandidateState.CANCELLED
            and current in {
                CandidateState.STAGED,
                CandidateState.STATICALLY_VALIDATED,
            }
            else (
                CandidateState.REJECTED
                if terminal_state == CandidateState.REJECTED
                else CandidateState.FAILED
            )
        )
        next_session = working_session.transition_candidate(
            candidate_plan.candidate_id,
            allowed_terminal,
            evidence_refs=evidence_ref,
            reason=failure_reason or "candidate failed prebenchmark validation",
        )
        status = allowed_terminal.value
    return CandidatePrebenchmarkOutcome(
        status=status,
        plan=validation_plan,
        record=record,
        sanity_run=sanity_run,
        evidence_path=evidence_path,
        session=next_session,
    )


def validate_benchmarked_candidate(
    *,
    plan: ValidationPlan,
    prebenchmark_record: ContractDocument,
    benchmark_record: ContractDocument,
    session: OptimizationSession,
    adapter: BehavioralValidationAdapter,
    workspace_root: Path,
    cancel_requested: Callable[[], bool] | None = None,
) -> CandidateValidationOutcome:
    _require_state(
        session,
        plan.candidate_id,
        CandidateState.BENCHMARKED,
        "candidate must complete matched benchmarking before full validation",
    )
    if (
        prebenchmark_record.schema != PREBENCHMARK_RECORD_SCHEMA
        or prebenchmark_record.to_json()["status"] != "passed"
        or prebenchmark_record.to_json()["candidate_id"] != plan.candidate_id
        or benchmark_record.schema != BENCHMARK_RECORD_SCHEMA
        or benchmark_record.to_json()["candidate_id"] != plan.candidate_id
    ):
        raise ModelCompileError(
            "full validation evidence does not match a prevalidated candidate"
        )
    runs: list[ValidationRun] = []
    product_performance = None
    failure_reason: str | None = None
    terminal_state = CandidateState.REJECTED
    benchmark = benchmark_record.to_json()
    try:
        if benchmark["decision"] != "materially_faster":
            failure_reason = (
                "candidate did not run faster than the matched exact "
                "implementation"
            )
        else:
            local = execute_validation_stage(
                plan,
                stage="full_local",
                adapter=adapter,
                cancel_requested=cancel_requested,
            )
            runs.append(local)
            if local.status != "completed":
                failure_reason = (
                    "candidate failed full local behavioral validation"
                )
                if local.status == "cancelled":
                    terminal_state = CandidateState.CANCELLED
            else:
                whole = execute_validation_stage(
                    plan,
                    stage="whole_model",
                    adapter=adapter,
                    cancel_requested=cancel_requested,
                )
                runs.append(whole)
                if whole.status != "completed":
                    failure_reason = (
                        "candidate failed whole-model free-running validation"
                    )
                    if whole.status == "cancelled":
                        terminal_state = CandidateState.CANCELLED
                else:
                    product_performance = (
                        qualify_whole_model_product_performance(whole)
                    )
                    if product_performance["status"] != "passed":
                        failure_reason = str(
                            product_performance["reason"]
                        )
    except ModelCompileCancelled as error:
        failure_reason = str(error)
        terminal_state = CandidateState.CANCELLED
    except Exception as error:
        failure_reason = str(error)
        terminal_state = CandidateState.FAILED

    if product_performance is None:
        whole = next(
            (
                run
                for run in runs
                if run.to_json()["stage"] == "whole_model"
            ),
            None,
        )
        if whole is None or whole.status != "completed":
            product_performance = (
                qualify_whole_model_product_performance(whole)
            )
        else:
            product_performance = {
                "status": "failed",
                "reason": (
                    failure_reason
                    or "whole-model product performance evidence is invalid"
                ),
                "metrics": {},
            }
    record = build_validation_record(
        plan=plan,
        prebenchmark_record=prebenchmark_record,
        benchmark_record=benchmark_record,
        runs=tuple(runs),
        product_performance=product_performance,
        failure_reason=failure_reason,
    )
    evidence_path = publish_validation_evidence(
        workspace_root,
        plan=plan,
        prebenchmark_record=prebenchmark_record,
        benchmark_record=benchmark_record,
        runs=tuple(runs),
        record=record,
        artifact_source=adapter,
    )
    evidence_ref = (
        f"validations/{record.to_json()['validation_id']}/record.json",
    )
    if record.to_json()["status"] == "passed":
        next_state = CandidateState.BEHAVIORALLY_VALIDATED
        reason = (
            "material local speedup, full local checks, whole-model "
            "free-running validation, and warmed product performance all passed"
        )
        status = "passed"
    else:
        next_state = terminal_state
        reason = failure_reason or "candidate failed behavioral validation"
        status = next_state.value
    next_session = session.transition_candidate(
        plan.candidate_id,
        next_state,
        evidence_refs=evidence_ref,
        reason=reason,
    )
    return CandidateValidationOutcome(
        status=status,
        plan=plan,
        record=record,
        runs=tuple(runs),
        evidence_path=evidence_path,
        session=next_session,
    )


def _validate_static_candidate(
    *,
    package_dir: Path,
    candidate_workspace_root: Path,
    candidate_plan: ProviderCandidatePlan,
    construction_record: ContractDocument,
) -> dict:
    loaded = load_staged_candidate(
        candidate_workspace_root,
        candidate_plan.candidate_id,
        package_dir=package_dir,
    )
    if (
        loaded.record != construction_record
        or loaded.build_plan != candidate_plan.construction_requirements
    ):
        raise ModelCompileError(
            "staged candidate does not match construction evidence"
        )
    candidate_document = ContractDocument.from_json(
        read_json(loaded.path / "contracts" / "candidate.json"),
        expected_schema=REPRESENTATION_CANDIDATE_SCHEMA,
    )
    if candidate_document != candidate_plan.candidate:
        raise ModelCompileError(
            "staged representation candidate contract changed"
        )
    manifest = validate_staged_candidate(
        loaded.path,
        expected_candidate_id=candidate_plan.candidate_id,
        expected_build_plan=candidate_plan.construction_requirements,
    )
    evidence = integrity_evidence(manifest)
    return {
        "status": "passed",
        "staged_integrity_digest": evidence["digest"],
        "artifact_count": evidence["file_count"],
    }


def _validate_plan_inputs(
    candidate_plan: ProviderCandidatePlan,
    construction_record: ContractDocument,
    validation_plan: ValidationPlan,
) -> None:
    if (
        validation_plan.candidate_id != candidate_plan.candidate_id
        or validation_plan.to_json()["construction_record_digest"]
        != construction_record.digest
        or validation_plan.behavioral_contract
        != candidate_plan.candidate.to_json()["behavioral_contract"]
    ):
        raise ModelCompileError(
            "validation plan is not bound to the staged candidate"
        )


def _candidate_state(
    session: OptimizationSession,
    candidate_id: str,
) -> CandidateState:
    matching = [
        candidate
        for candidate in session.candidates
        if candidate.candidate_id == candidate_id
    ]
    if len(matching) != 1:
        raise ModelCompileError(
            "optimization session does not contain exactly one candidate"
        )
    return matching[0].state


def _require_state(
    session: OptimizationSession,
    candidate_id: str,
    expected: CandidateState,
    message: str,
) -> None:
    if _candidate_state(session, candidate_id) != expected:
        raise ModelCompileError(message)
