from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable, Iterable

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.analysis.context import AnalysisBudget
from nerve.representation_optimizer.analysis.engine import analyze_scope
from nerve.representation_optimizer.automation.budgets import BudgetLedger
from nerve.representation_optimizer.automation.contracts import (
    OPTIMIZER_RUN_SCHEMA,
    OptimizationBudget,
)
from nerve.representation_optimizer.automation.events import EventJournal
from nerve.representation_optimizer.automation.pipeline import execute_candidate
from nerve.representation_optimizer.automation.records import (
    build_provider_records,
    error_document,
    finish_candidate,
    new_candidate_record,
    record_candidate_failure,
)
from nerve.representation_optimizer.automation.report import (
    AutomatedOptimizationOutcome,
    build_report,
    publish_report,
)
from nerve.representation_optimizer.automation.storage import (
    relative_ref,
    write_new_json,
)
from nerve.representation_optimizer.automation.target import OptimizationTarget
from nerve.representation_optimizer.contracts import (
    representation_candidate_equivalence_key,
    stable_contract_id,
)
from nerve.representation_optimizer.lifecycle import CandidateState, OptimizationSession
from nerve.representation_optimizer.promotion.orchestrator import PreparedPromotion
from nerve.representation_optimizer.promotion.publication import (
    publish_promoted_package,
)
from nerve.representation_optimizer.providers.registry import ProviderRegistry
from nerve.representation_optimizer.providers.types import ProviderProblem
from nerve.representation_optimizer.scope_enumeration.catalog import (
    load_optimization_scope_catalog,
)
from nerve.representation_optimizer.stage import load_optimizer_stage


def run_automated_optimizer(
    *,
    package_dir: Path,
    output_package_dir: Path,
    run_root: Path,
    providers: ProviderRegistry,
    targets: Iterable[OptimizationTarget],
    budget: OptimizationBudget,
    analysis_budget: AnalysisBudget | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> AutomatedOptimizationOutcome:
    source = package_dir.resolve()
    output = output_package_dir.resolve()
    run_root = run_root.resolve()
    _validate_fresh_paths(source, output, run_root)
    targets = tuple(sorted(targets, key=lambda target: target.target_id))
    if not targets or len({target.target_id for target in targets}) != len(targets):
        raise ModelCompileError(
            "automated optimizer targets must be non-empty and unique"
        )
    stage = load_optimizer_stage(
        source / "optimization" / "stage.json",
        package_dir=source,
    )
    catalog = load_optimization_scope_catalog(
        source / "optimization" / "scopes.json"
    )
    session = OptimizationSession.from_json(stage["session"])
    run_id = stable_contract_id(
        "automated_optimizer_run",
        catalog["package_id"],
        stage["exact_baseline"]["contract_digest"],
        budget.to_json(),
        [target.target_id for target in targets],
    )
    run_root.mkdir(parents=True, exist_ok=False)
    run_manifest = {
        "schema": OPTIMIZER_RUN_SCHEMA,
        "run_id": run_id,
        "package_id": catalog["package_id"],
        "source_package": str(source),
        "requested_output_package": str(output),
        "exact_baseline_digest": stage["exact_baseline"]["contract_digest"],
        "target_ids": [target.target_id for target in targets],
        "budget": budget.to_json(),
    }
    write_new_json(run_root / "run.json", run_manifest)
    journal = EventJournal(run_root / "events.jsonl")
    ledger = BudgetLedger(budget)
    scope_records: list[Json] = []
    provider_records: list[Json] = []
    duplicate_records: list[Json] = []
    candidate_records: dict[str, Json] = {}
    promotions: list[PreparedPromotion] = []
    promotion_records: list[Json] = []
    seen_equivalence: dict[str, str] = {}
    analysis_directories: dict[str, Path] = {}
    publication: Json = {"status": "not_attempted", "reason": "no candidates promoted"}
    fatal_error: Exception | None = None

    journal.record(
        phase="run",
        status="started",
        details={"run_id": run_id, "target_ids": [item.target_id for item in targets]},
    )
    try:
        scopes = tuple(sorted(catalog["scopes"], key=lambda item: item["scope_id"]))
        contracts = {
            str(item["scope_id"]): item for item in catalog["source_contracts"]
        }
        for scope in scopes:
            scope_id = str(scope["scope_id"])
            admitted, reason = ledger.admit_scope()
            if not admitted:
                record = {
                    "scope_id": scope_id,
                    "kind": scope["kind"],
                    "status": "budget_skipped",
                    "reason": reason,
                    "analysis_ref": None,
                    "structures": [],
                }
                scope_records.append(record)
                journal.record(
                    phase="analysis",
                    status="budget_skipped",
                    scope_id=scope_id,
                    details={"reason": reason},
                )
                continue
            analysis_directory = run_root / "analysis" / scope_id
            try:
                analysis = analyze_scope(
                    package_dir=source,
                    scope_id=scope_id,
                    budget=analysis_budget,
                    output_dir=analysis_directory,
                )
            except Exception as error:
                record = {
                    "scope_id": scope_id,
                    "kind": scope["kind"],
                    "status": "failed",
                    "reason": str(error),
                    "analysis_ref": None,
                    "structures": [],
                }
                scope_records.append(record)
                journal.record(
                    phase="analysis",
                    status="failed",
                    scope_id=scope_id,
                    details=error_document(error),
                )
                continue
            for evidence in analysis.evidence:
                analysis_directories[str(evidence["evidence_id"])] = analysis_directory
            analysis_ref = relative_ref(run_root, analysis_directory)
            structures = [
                {
                    "evidence_id": evidence["evidence_id"],
                    "analyzer": dict(evidence["analyzer"]),
                    "claims": deepcopy(evidence["claims"]),
                }
                for evidence in analysis.evidence
                if evidence["claims"]
            ]
            scope_records.append(
                {
                    "scope_id": scope_id,
                    "kind": scope["kind"],
                    "status": "analyzed",
                    "reason": "all selected analyzers completed",
                    "analysis_ref": analysis_ref,
                    "structures": structures,
                }
            )
            journal.record(
                phase="analysis",
                status="completed",
                scope_id=scope_id,
                evidence_refs=(analysis_ref,),
                details={"structure_record_count": len(structures)},
            )
            source_contract = contracts[scope_id]
            for target in targets:
                problem = ProviderProblem.from_documents(
                    package_id=str(catalog["package_id"]),
                    scopes=(scope,),
                    source_contracts=(source_contract,),
                    evidence=analysis.evidence,
                    hardware_profile=target.synthesis_profile,
                )
                registry_report = providers.run(problem)
                evaluations = build_provider_records(
                    scope_id=scope_id,
                    target_id=target.target_id,
                    report=registry_report,
                )
                provider_records.extend(evaluations)
                for evaluation in evaluations:
                    journal.record(
                        phase="provider_evaluation",
                        status=str(evaluation["status"]),
                        scope_id=scope_id,
                        target_id=target.target_id,
                        details={
                            "provider": evaluation["provider"],
                            "candidate_ids": evaluation["candidate_ids"],
                            "error": evaluation["error"],
                        },
                    )
                for duplicate in registry_report.duplicate_candidates:
                    record = {
                        "scope_id": scope_id,
                        "target_id": target.target_id,
                        **deepcopy(duplicate),
                    }
                    duplicate_records.append(record)
                    journal.record(
                        phase="synthesis",
                        status="deduplicated",
                        scope_id=scope_id,
                        target_id=target.target_id,
                        candidate_id=str(record["discarded_candidate_id"]),
                        details={
                            "kept_candidate_id": record["kept_candidate_id"],
                            "equivalence_key": record["equivalence_key"],
                        },
                    )
                for plan in registry_report.candidates:
                    equivalence = representation_candidate_equivalence_key(
                        plan.candidate.to_json()
                    )
                    kept = seen_equivalence.get(equivalence)
                    if kept is not None:
                        duplicate_records.append(
                            {
                                "scope_id": scope_id,
                                "target_id": target.target_id,
                                "equivalence_key": equivalence,
                                "kept_candidate_id": kept,
                                "discarded_candidate_id": plan.candidate_id,
                                "reason": "duplicate across optimization problems",
                            }
                        )
                        journal.record(
                            phase="synthesis",
                            status="deduplicated",
                            scope_id=scope_id,
                            target_id=target.target_id,
                            candidate_id=plan.candidate_id,
                            details={"kept_candidate_id": kept},
                        )
                        continue
                    seen_equivalence[equivalence] = plan.candidate_id
                    session = session.register_candidate(
                        plan.candidate_id,
                        tuple(plan.candidate.to_json()["source_contract_digests"]),
                    )
                    candidate_records[plan.candidate_id] = new_candidate_record(
                        plan, scope_id=scope_id, target_id=target.target_id
                    )
                    try:
                        execution_cost = target.estimate_execution_nanoseconds(
                            plan, target.benchmark_policy
                        )
                        admission = ledger.admit_candidate(
                            plan,
                            execution_nanoseconds=execution_cost,
                        )
                        budget_path = write_new_json(
                            run_root
                            / "decisions"
                            / plan.candidate_id
                            / "budget.json",
                            admission.to_json(candidate_id=plan.candidate_id),
                        )
                        budget_ref = relative_ref(run_root, budget_path)
                        candidate_records[plan.candidate_id]["budget_decision_ref"] = (
                            budget_ref
                        )
                        if not admission.admitted:
                            session = session.transition_candidate(
                                plan.candidate_id,
                                CandidateState.REJECTED,
                                evidence_refs=(budget_ref,),
                                reason="; ".join(admission.reasons),
                            )
                            finish_candidate(
                                candidate_records[plan.candidate_id],
                                session,
                                plan.candidate_id,
                                rejection_reasons=list(admission.reasons),
                            )
                            journal.record(
                                phase="budget",
                                status="rejected",
                                scope_id=scope_id,
                                target_id=target.target_id,
                                candidate_id=plan.candidate_id,
                                evidence_refs=(budget_ref,),
                                details={"reasons": list(admission.reasons)},
                            )
                            continue
                        (
                            session,
                            promotion,
                            candidate_updates,
                        ) = execute_candidate(
                            source=source,
                            run_root=run_root,
                            plan=plan,
                            source_contract=source_contract,
                            target=target,
                            session=session,
                            analysis_directories=analysis_directories,
                            journal=journal,
                            cancel_requested=cancel_requested,
                        )
                        candidate_records[plan.candidate_id].update(candidate_updates)
                        finish_candidate(
                            candidate_records[plan.candidate_id],
                            session,
                            plan.candidate_id,
                        )
                        if promotion is not None:
                            promotions.append(promotion)
                            promotion_records.append(
                                {
                                    "candidate_id": plan.candidate_id,
                                    "implementation_id": promotion.implementation_id,
                                    "promotion_id": promotion.decision.to_json()[
                                        "promotion_id"
                                    ],
                                    "reason": promotion.decision.to_json()["reason"],
                                }
                            )
                    except Exception as error:
                        session = record_candidate_failure(
                            run_root=run_root,
                            plan=plan,
                            session=session,
                            error=error,
                            journal=journal,
                            scope_id=scope_id,
                            target_id=target.target_id,
                        )
                        record = candidate_records[plan.candidate_id]
                        record["failure"] = error_document(error)
                        finish_candidate(
                            record,
                            session,
                            plan.candidate_id,
                            rejection_reasons=[str(error)],
                        )
        if promotions:
            ordered = tuple(sorted(promotions, key=lambda item: item.implementation_id))
            publication_path = publish_promoted_package(
                source_package_dir=source,
                destination_package_dir=output,
                promotions=ordered,
                session=session,
            )
            published_stage = load_optimizer_stage(
                publication_path / "optimization" / "stage.json",
                package_dir=publication_path,
            )
            session = OptimizationSession.from_json(published_stage["session"])
            for promotion in ordered:
                finish_candidate(
                    candidate_records[promotion.candidate_plan.candidate_id],
                    session,
                    promotion.candidate_plan.candidate_id,
                )
            publication = {
                "status": "published",
                "output_package": str(publication_path),
                "implementation_ids": [
                    promotion.implementation_id for promotion in ordered
                ],
            }
            journal.record(
                phase="publication",
                status="completed",
                details=publication,
            )
            status = "completed"
            result_package = publication_path
        else:
            status = "completed_no_changes"
            result_package = source
            publication = {
                "status": "not_required",
                "reason": "no candidate was both materially faster and fully valid",
            }
        journal.record(
            phase="run",
            status=status,
            details={"promotion_count": len(promotions)},
        )
    except Exception as error:
        fatal_error = error
        status = "failed"
        result_package = source
        publication = {
            "status": "failed",
            "reason": str(error),
            "error": error_document(error),
        }
        journal.record(
            phase="run",
            status="failed",
            details=error_document(error),
        )

    report = build_report(
        run_id=run_id,
        package_id=str(catalog["package_id"]),
        source_package=source,
        output_package=result_package,
        status=status,
        budget=budget.to_json(),
        budget_usage=ledger.usage.to_json(),
        scopes=scope_records,
        provider_evaluations=provider_records,
        duplicate_candidates=duplicate_records,
        candidates=[
            candidate_records[candidate_id]
            for candidate_id in sorted(candidate_records)
        ],
        promotions=sorted(
            promotion_records, key=lambda item: item["implementation_id"]
        ),
        publication=publication,
        session=session,
        event_journal_ref="events.jsonl",
        event_count=journal.event_count,
    )
    report_path = publish_report(run_root, report)
    if fatal_error is not None:
        raise ModelCompileError(
            f"automated optimizer failed safely; report: {report_path}"
        ) from fatal_error
    return AutomatedOptimizationOutcome(
        report_path=report_path,
        report=report,
        output_package_dir=result_package,
        session=session,
    )


def _validate_fresh_paths(source: Path, output: Path, run_root: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        raise ModelCompileError("automated optimizer source package is invalid")
    if output == source or output.exists() or output.is_symlink():
        raise ModelCompileError(
            "automated optimizer output must be a fresh path distinct from source"
        )
    if run_root.exists() or run_root.is_symlink():
        raise ModelCompileError("automated optimizer run root must be fresh")
    for candidate in (output, run_root):
        try:
            candidate.relative_to(source)
        except ValueError:
            continue
        raise ModelCompileError(
            "automated optimizer output and workspaces must stay outside source package"
        )
    if _contains(output, run_root) or _contains(run_root, output):
        raise ModelCompileError(
            "automated optimizer output package and run root must not overlap"
        )


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True
