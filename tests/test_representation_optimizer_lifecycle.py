from __future__ import annotations

import pytest

from nerve.representation_optimizer.contracts import (
    ContractValidationError,
    contract_digest,
)
from nerve.representation_optimizer.lifecycle import (
    CandidateLifecycle,
    CandidateState,
    OptimizationSession,
)


def test_candidate_lifecycle_requires_ordered_evidence_carrying_transitions() -> None:
    lifecycle = CandidateLifecycle.create("candidate_a", (contract_digest("source"),))

    with pytest.raises(ContractValidationError, match="cannot transition"):
        lifecycle.transition(
            CandidateState.BENCHMARKED,
            evidence_refs=("benchmark.json",),
            reason="skipped staging",
        )
    lifecycle = lifecycle.transition(
        CandidateState.STAGED,
        evidence_refs=("construction.json",),
        reason="candidate was constructed in isolation",
    )
    with pytest.raises(ContractValidationError, match="requires evidence"):
        lifecycle.transition(
            CandidateState.STATICALLY_VALIDATED,
            evidence_refs=(),
            reason="no evidence",
        )
    lifecycle = lifecycle.transition(
        CandidateState.STATICALLY_VALIDATED,
        evidence_refs=("static_validation.json",),
        reason="all static contracts passed",
    )

    assert CandidateLifecycle.from_json(lifecycle.to_json()) == lifecycle


def test_candidate_failure_is_isolated_and_exact_baseline_is_immutable() -> None:
    baseline_digest = contract_digest({"exact": "lowered graph"})
    source_digest = contract_digest({"source": "scope"})
    session = OptimizationSession.create("fixture_package", baseline_digest)
    session = session.register_candidate("candidate_a", (source_digest,))
    session = session.register_candidate("candidate_b", (source_digest,))

    session = session.transition_candidate(
        "candidate_a",
        CandidateState.FAILED,
        evidence_refs=("diagnostics/candidate_a.json",),
        reason="provider raised an isolated construction error",
    )
    session = session.transition_candidate(
        "candidate_b",
        CandidateState.STAGED,
        evidence_refs=("construction/candidate_b.json",),
        reason="unrelated candidate constructed successfully",
    )

    by_id = {candidate.candidate_id: candidate for candidate in session.candidates}
    assert by_id["candidate_a"].state == CandidateState.FAILED
    assert by_id["candidate_b"].state == CandidateState.STAGED
    assert session.exact_baseline_digest == baseline_digest
    assert OptimizationSession.from_json(session.to_json()) == session


def test_session_rejects_duplicate_candidates_and_corrupt_history() -> None:
    session = OptimizationSession.create("fixture_package", contract_digest("baseline"))
    session = session.register_candidate("candidate_a", (contract_digest("source"),))

    with pytest.raises(ContractValidationError, match="already registered"):
        session.register_candidate("candidate_a", (contract_digest("source"),))

    corrupted = session.to_json()
    corrupted["candidates"][0]["state"] = "published"
    with pytest.raises(ContractValidationError, match="does not match"):
        OptimizationSession.from_json(corrupted)
