from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from nerve.compilation import ModelCompileCancelled, ModelCompileError
from nerve.representation_optimizer.benchmarking.contracts import (
    BenchmarkPlan,
    BenchmarkRun,
)
from nerve.representation_optimizer.benchmarking.protocols import (
    NormalExecutionAdapter,
)
from nerve.representation_optimizer.benchmarking.runner import (
    execute_benchmark_plan,
)
from nerve.representation_optimizer.benchmarking.statistics import (
    summarize_benchmark,
)
from nerve.representation_optimizer.benchmarking.storage import (
    publish_benchmark_evidence,
)
from nerve.representation_optimizer.contracts import ContractDocument
from nerve.representation_optimizer.lifecycle import (
    CandidateState,
    OptimizationSession,
)


@dataclass(frozen=True)
class CandidateBenchmarkOutcome:
    plan: BenchmarkPlan
    run: BenchmarkRun
    record: ContractDocument
    evidence_path: Path
    session: OptimizationSession


def benchmark_candidate(
    *,
    plan: BenchmarkPlan,
    construction_record: ContractDocument,
    session: OptimizationSession,
    adapter: NormalExecutionAdapter,
    workspace_root: Path,
    cancel_requested: Callable[[], bool] | None = None,
) -> CandidateBenchmarkOutcome:
    _validate_session(plan, construction_record, session)
    run = execute_benchmark_plan(
        plan,
        adapter,
        cancel_requested=cancel_requested,
    )
    run_status = run.to_json()["status"]
    if run_status == "cancelled":
        raise ModelCompileCancelled("matched candidate benchmark was cancelled")
    if run_status != "completed":
        raise ModelCompileError(
            "candidate benchmark did not complete all matched observations"
        )
    record = summarize_benchmark(
        plan=plan,
        run=run,
        construction_record=construction_record,
    )
    evidence_path = publish_benchmark_evidence(
        workspace_root,
        plan=plan,
        run=run,
        record=record,
        trace_source=adapter,
    )
    evidence_ref = (
        f"benchmarks/{record.to_json()['benchmark_id']}/record.json",
    )
    next_session = session.transition_candidate(
        plan.candidate_id,
        CandidateState.BENCHMARKED,
        evidence_refs=evidence_ref,
        reason=(
            "candidate and exact reference completed the matched benchmark plan"
        ),
    )
    return CandidateBenchmarkOutcome(
        plan=plan,
        run=run,
        record=record,
        evidence_path=evidence_path,
        session=next_session,
    )


def _validate_session(
    plan: BenchmarkPlan,
    construction_record: ContractDocument,
    session: OptimizationSession,
) -> None:
    if construction_record.digest != plan.to_json()[
        "construction_record_digest"
    ]:
        raise ModelCompileError(
            "benchmark plan does not reference candidate construction evidence"
        )
    matching = [
        candidate
        for candidate in session.candidates
        if candidate.candidate_id == plan.candidate_id
    ]
    if (
        len(matching) != 1
        or matching[0].state
        != CandidateState.PREBENCHMARK_VALIDATED
    ):
        raise ModelCompileError(
            "candidate must pass proof and prebenchmark behavioral sanity "
            "before benchmarking"
        )
