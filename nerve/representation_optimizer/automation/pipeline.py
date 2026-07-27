from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable

from nerve.compilation import (
    Json,
    ModelCompileCancelled,
    ModelCompileError,
    check_compile_cancelled,
)
from nerve.representation_optimizer.automation.events import EventJournal
from nerve.representation_optimizer.automation.target import OptimizationTarget
from nerve.representation_optimizer.benchmarking.orchestrator import (
    benchmark_candidate,
)
from nerve.representation_optimizer.benchmarking.planning import (
    build_benchmark_plan,
)
from nerve.representation_optimizer.lifecycle import (
    CandidateState,
    OptimizationSession,
    TERMINAL_CANDIDATE_STATES,
)
from nerve.representation_optimizer.promotion.orchestrator import (
    PreparedPromotion,
    prepare_candidate_promotion,
)
from nerve.representation_optimizer.providers.types import ProviderCandidatePlan
from nerve.representation_optimizer.staging.contracts import staged_file_digest
from nerve.representation_optimizer.staging.orchestrator import stage_candidate
from nerve.representation_optimizer.validation.orchestrator import (
    prepare_candidate_for_benchmark,
    validate_benchmarked_candidate,
)
from nerve.representation_optimizer.validation.planning import (
    build_validation_plan,
)


def execute_candidate(
    *,
    source: Path,
    run_root: Path,
    plan: ProviderCandidatePlan,
    source_contract: Json,
    target: OptimizationTarget,
    session: OptimizationSession,
    analysis_directories: dict[str, Path],
    journal: EventJournal,
    cancel_requested: Callable[[], bool] | None,
) -> tuple[OptimizationSession, PreparedPromotion | None, Json]:
    candidate_id = plan.candidate_id
    updates: Json = {}
    cancelled = _cancelled_session(
        session,
        candidate_id,
        cancel_requested=cancel_requested,
        reason="candidate execution was cancelled before construction",
    )
    if cancelled is not None:
        return cancelled, None, updates
    candidate_workspace = run_root / "workspaces" / "candidates"
    benchmark_workspace = run_root / "workspaces" / "benchmark"
    validation_workspace = run_root / "workspaces" / "validation"
    toolchain = target.toolchains.resolve(plan)
    construction = stage_candidate(
        package_dir=source,
        workspace_root=candidate_workspace,
        plan=plan,
        session=session,
        semantic_constructor=toolchain.semantic_constructor,
        ordinary_relowerer=toolchain.ordinary_relowerer,
        physical_optimizer=toolchain.physical_optimizer,
        artifact_validators=toolchain.artifact_validators,
        cancel_requested=cancel_requested,
    )
    updates = {
        "construction_status": construction.status,
        "construction_record_digest": construction.record.digest,
        "construction_diagnostics": deepcopy(
            construction.record.to_json()["diagnostics"]
        ),
    }
    journal.record(
        phase="construction",
        status=construction.status,
        target_id=target.target_id,
        candidate_id=candidate_id,
        evidence_refs=(
            f"workspaces/candidates/records/"
            f"{construction.record.to_json()['construction_id']}.json",
        ),
    )
    if construction.status != "completed":
        return construction.session, None, updates
    session = construction.session
    cancelled = _cancelled_session(
        session,
        candidate_id,
        cancel_requested=cancel_requested,
        reason="candidate execution was cancelled after construction",
    )
    if cancelled is not None:
        return cancelled, None, updates
    benchmark_plan = build_benchmark_plan(
        candidate_plan=plan,
        construction_record=construction.record,
        hardware_profiles=target.hardware_profiles,
        reference_implementation_id=source_contract["exact_reference"][
            "implementation_id"
        ],
        reference_contract_digest=source_contract["contract_digest"],
        reference_artifact_refs=_reference_artifacts(source, source_contract),
        matched_conditions=target.matched_conditions,
        policy=target.benchmark_policy,
    )
    validation_plan = build_validation_plan(
        candidate_plan=plan,
        construction_record=construction.record,
        benchmark_plan=benchmark_plan,
    )
    with target.lease_manager.acquire(target):
        prebenchmark = prepare_candidate_for_benchmark(
            package_dir=source,
            candidate_workspace_root=candidate_workspace,
            validation_workspace_root=validation_workspace,
            candidate_plan=plan,
            construction_record=construction.record,
            validation_plan=validation_plan,
            session=session,
            proof_verifiers=target.proof_verifiers,
            adapter=target.validation_adapter,
            cancel_requested=cancel_requested,
        )
    session = prebenchmark.session
    updates["prebenchmark_status"] = prebenchmark.status
    updates["prebenchmark_record_digest"] = prebenchmark.record.digest
    journal.record(
        phase="prebenchmark_validation",
        status=prebenchmark.status,
        target_id=target.target_id,
        candidate_id=candidate_id,
        evidence_refs=(
            f"workspaces/validation/prebenchmark/"
            f"{prebenchmark.record.to_json()['prebenchmark_id']}/record.json",
        ),
    )
    if prebenchmark.status != "passed":
        return session, None, updates
    cancelled = _cancelled_session(
        session,
        candidate_id,
        cancel_requested=cancel_requested,
        reason="candidate execution was cancelled before benchmarking",
    )
    if cancelled is not None:
        return cancelled, None, updates
    try:
        with target.lease_manager.acquire(target):
            benchmark = benchmark_candidate(
                plan=benchmark_plan,
                construction_record=construction.record,
                session=session,
                adapter=target.benchmark_adapter,
                workspace_root=benchmark_workspace,
                cancel_requested=cancel_requested,
            )
    except ModelCompileCancelled as error:
        session = _cancel_candidate(
            session,
            candidate_id,
            reason=str(error),
        )
        journal.record(
            phase="benchmark",
            status="cancelled",
            target_id=target.target_id,
            candidate_id=candidate_id,
            details={"type": type(error).__name__, "message": str(error)},
        )
        return session, None, updates
    session = benchmark.session
    decision = str(benchmark.record.to_json()["decision"])
    updates["benchmark_decision"] = decision
    updates["benchmark_record_digest"] = benchmark.record.digest
    journal.record(
        phase="benchmark",
        status=decision,
        target_id=target.target_id,
        candidate_id=candidate_id,
        evidence_refs=(
            f"workspaces/benchmark/benchmarks/"
            f"{benchmark.record.to_json()['benchmark_id']}/record.json",
        ),
    )
    if decision == "materially_faster":
        cancelled = _cancelled_session(
            session,
            candidate_id,
            cancel_requested=cancel_requested,
            reason="candidate execution was cancelled before full validation",
        )
        if cancelled is not None:
            return cancelled, None, updates
        with target.lease_manager.acquire(target):
            validation = validate_benchmarked_candidate(
                plan=validation_plan,
                prebenchmark_record=prebenchmark.record,
                benchmark_record=benchmark.record,
                session=session,
                adapter=target.validation_adapter,
                workspace_root=validation_workspace,
                cancel_requested=cancel_requested,
            )
    else:
        # The validation orchestrator records an auditable performance
        # rejection without opening an execution session. Do not acquire or
        # probe accelerators for a candidate that cannot be promoted.
        validation = validate_benchmarked_candidate(
            plan=validation_plan,
            prebenchmark_record=prebenchmark.record,
            benchmark_record=benchmark.record,
            session=session,
            adapter=target.validation_adapter,
            workspace_root=validation_workspace,
            cancel_requested=cancel_requested,
        )
    session = validation.session
    updates["validation_status"] = validation.status
    updates["validation_record_digest"] = validation.record.digest
    updates["counterexamples"] = deepcopy(
        validation.record.to_json().get("counterexamples", [])
    )
    journal.record(
        phase="full_validation",
        status=validation.status,
        target_id=target.target_id,
        candidate_id=candidate_id,
        evidence_refs=(
            f"workspaces/validation/validations/"
            f"{validation.record.to_json()['validation_id']}/record.json",
        ),
    )
    if validation.status != "passed":
        return session, None, updates
    cancelled = _cancelled_session(
        session,
        candidate_id,
        cancel_requested=cancel_requested,
        reason="candidate execution was cancelled before promotion",
    )
    if cancelled is not None:
        return cancelled, None, updates
    evidence_ids = plan.candidate.to_json()["evidence_refs"]
    directories = tuple(
        sorted(
            {analysis_directories[evidence_id] for evidence_id in evidence_ids},
            key=lambda path: path.as_posix(),
        )
    )
    promotion = prepare_candidate_promotion(
        package_dir=source,
        candidate_workspace_root=candidate_workspace,
        benchmark_workspace_root=benchmark_workspace,
        validation_workspace_root=validation_workspace,
        analysis_run_directories=directories,
        candidate_plan=plan,
        construction_record=construction.record,
        benchmark_record=benchmark.record,
        validation_record=validation.record,
        hardware_profiles=target.hardware_profiles,
        session=session,
        reason=(
            "candidate won every complete matched workload and passed proof, "
            "sanity, full-local, and whole-model behavioral validation"
        ),
    )
    session = promotion.session
    updates["promotion_id"] = promotion.decision.to_json()["promotion_id"]
    journal.record(
        phase="promotion",
        status="prepared",
        target_id=target.target_id,
        candidate_id=candidate_id,
        evidence_refs=(
            f"promotions/{promotion.decision.to_json()['promotion_id']}.json",
        ),
        details={"implementation_id": promotion.implementation_id},
    )
    cancelled = _cancelled_session(
        session,
        candidate_id,
        cancel_requested=cancel_requested,
        reason="candidate execution was cancelled after promotion preparation",
    )
    if cancelled is not None:
        updates["promotion_id"] = None
        return cancelled, None, updates
    return session, promotion, updates


def _reference_artifacts(package_dir: Path, source_contract: Json) -> tuple[Json, ...]:
    references = []
    for value in source_contract["exact_reference"]["artifact_refs"]:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ModelCompileError("exact reference artifact escapes package")
        references.append(
            {
                "path": relative.as_posix(),
                "digest": staged_file_digest(package_dir / relative),
            }
        )
    return tuple(sorted(references, key=lambda item: item["path"]))


def _cancelled_session(
    session: OptimizationSession,
    candidate_id: str,
    *,
    cancel_requested: Callable[[], bool] | None,
    reason: str,
) -> OptimizationSession | None:
    try:
        check_compile_cancelled(cancel_requested)
    except ModelCompileCancelled:
        return _cancel_candidate(session, candidate_id, reason=reason)
    return None


def _cancel_candidate(
    session: OptimizationSession,
    candidate_id: str,
    *,
    reason: str,
) -> OptimizationSession:
    matches = [
        candidate
        for candidate in session.candidates
        if candidate.candidate_id == candidate_id
    ]
    if len(matches) != 1:
        raise ModelCompileError(
            "candidate cancellation requires one registered lifecycle"
        )
    lifecycle = matches[0]
    if lifecycle.state == CandidateState.CANCELLED:
        return session
    if lifecycle.state in TERMINAL_CANDIDATE_STATES:
        raise ModelCompileError(
            "candidate cancellation cannot rewrite a terminal lifecycle"
        )
    return session.transition_candidate(
        candidate_id,
        CandidateState.CANCELLED,
        evidence_refs=(),
        reason=reason,
    )
