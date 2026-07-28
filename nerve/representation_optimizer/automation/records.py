from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.automation.events import EventJournal
from nerve.representation_optimizer.automation.storage import (
    relative_ref,
    write_new_json,
)
from nerve.representation_optimizer.lifecycle import (
    CandidateState,
    OptimizationSession,
    TERMINAL_CANDIDATE_STATES,
)
from nerve.representation_optimizer.providers.types import ProviderCandidatePlan


def build_provider_records(
    *,
    scope_ids: tuple[str, ...],
    target_id: str,
    report,
) -> list[Json]:
    records = []
    for evaluation in report.evaluations:
        records.append(
            {
                "scope_ids": list(scope_ids),
                "target_id": target_id,
                "provider": evaluation.provider.to_json(),
                "descriptor_id": evaluation.descriptor_id,
                "status": evaluation.status,
                "semantic_match": _assessment(evaluation.semantic_match),
                "structural_match": _assessment(evaluation.structural_match),
                "evidence_assessment": (
                    None
                    if evaluation.evidence_assessment is None
                    else {
                        "accepted": evaluation.evidence_assessment.accepted,
                        "evidence_ids": list(
                            evaluation.evidence_assessment.evidence_ids
                        ),
                        "facts": deepcopy(evaluation.evidence_assessment.facts),
                        "reasons": list(evaluation.evidence_assessment.reasons),
                    }
                ),
                "candidate_ids": [
                    candidate.candidate_id for candidate in evaluation.candidates
                ],
                "error": deepcopy(evaluation.error),
            }
        )
    return records


def new_candidate_record(
    plan: ProviderCandidatePlan,
    *,
    scope_ids: tuple[str, ...],
    target_id: str,
) -> Json:
    candidate = plan.candidate.to_json()
    return {
        "candidate_id": plan.candidate_id,
        "scope_ids": list(scope_ids),
        "target_id": target_id,
        "provider": plan.provider.to_json(),
        "descriptor_id": candidate["descriptor_id"],
        "representation": deepcopy(candidate["representation"]),
        "status": CandidateState.SYNTHESIZED.value,
        "rejection_reasons": [],
        "budget_decision_ref": None,
        "construction_status": None,
        "construction_record_digest": None,
        "construction_diagnostics": [],
        "prebenchmark_status": None,
        "prebenchmark_record_digest": None,
        "benchmark_decision": None,
        "benchmark_record_digest": None,
        "validation_status": None,
        "validation_record_digest": None,
        "counterexamples": [],
        "promotion_id": None,
        "failure": None,
    }


def finish_candidate(
    record: Json,
    session: OptimizationSession,
    candidate_id: str,
    *,
    rejection_reasons: list[str] | None = None,
) -> None:
    lifecycle = candidate_lifecycle(session, candidate_id)
    record["status"] = lifecycle.state.value
    if rejection_reasons is not None:
        record["rejection_reasons"] = rejection_reasons
    elif lifecycle.state in {
        CandidateState.REJECTED,
        CandidateState.FAILED,
        CandidateState.CANCELLED,
    }:
        record["rejection_reasons"] = (
            list(record.get("construction_diagnostics", []))
            or ([lifecycle.history[-1]["reason"]] if lifecycle.history else [])
        )


def record_candidate_failure(
    *,
    run_root: Path,
    plan: ProviderCandidatePlan,
    session: OptimizationSession,
    error: Exception,
    journal: EventJournal,
    scope_id: str | None,
    target_id: str,
) -> OptimizationSession:
    lifecycle = candidate_lifecycle(session, plan.candidate_id)
    if lifecycle.state in TERMINAL_CANDIDATE_STATES:
        return session
    failure_path = write_new_json(
        run_root / "failures" / f"{plan.candidate_id}.json",
        {
            "schema": "nerve.optimizer.candidate_failure.v1",
            "candidate_id": plan.candidate_id,
            "state": lifecycle.state.value,
            "error": error_document(error),
        },
    )
    failure_ref = relative_ref(run_root, failure_path)
    next_session = session.transition_candidate(
        plan.candidate_id,
        CandidateState.FAILED,
        evidence_refs=(failure_ref,),
        reason=str(error) or type(error).__name__,
    )
    journal.record(
        phase="candidate",
        status="failed",
        scope_id=scope_id,
        target_id=target_id,
        candidate_id=plan.candidate_id,
        evidence_refs=(failure_ref,),
        details=error_document(error),
    )
    return next_session


def record_candidate_cancellation(
    *,
    plan: ProviderCandidatePlan,
    session: OptimizationSession,
    error: Exception,
    journal: EventJournal,
    scope_id: str | None,
    target_id: str,
) -> OptimizationSession:
    lifecycle = candidate_lifecycle(session, plan.candidate_id)
    next_session = session
    if lifecycle.state not in TERMINAL_CANDIDATE_STATES:
        next_session = session.transition_candidate(
            plan.candidate_id,
            CandidateState.CANCELLED,
            evidence_refs=(),
            reason=str(error) or "automated optimizer cancellation requested",
        )
    elif lifecycle.state != CandidateState.CANCELLED:
        raise ModelCompileError(
            "optimizer cancellation cannot rewrite a terminal candidate"
        )
    journal.record(
        phase="candidate",
        status="cancelled",
        scope_id=scope_id,
        target_id=target_id,
        candidate_id=plan.candidate_id,
        details=error_document(error),
    )
    return next_session


def candidate_lifecycle(
    session: OptimizationSession,
    candidate_id: str,
):
    matches = [
        candidate
        for candidate in session.candidates
        if candidate.candidate_id == candidate_id
    ]
    if len(matches) != 1:
        raise ModelCompileError(
            "automated optimizer session has no unique candidate lifecycle"
        )
    return matches[0]


def error_document(error: Exception) -> Json:
    return {"type": type(error).__name__, "message": str(error)}


def _assessment(value) -> Json | None:
    if value is None:
        return None
    return {
        "matched": value.matched,
        "reasons": list(value.reasons),
        "evidence_ids": list(value.evidence_ids),
    }
