from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.analysis.evidence import (
    validate_analysis_run_directory,
)
from nerve.representation_optimizer.automation.contracts import (
    OPTIMIZER_REPORT_SCHEMA,
    OPTIMIZER_RUN_SCHEMA,
)
from nerve.representation_optimizer.automation.events import read_event_journal
from nerve.representation_optimizer.automation.storage import (
    read_object,
    replace_json,
)
from nerve.representation_optimizer.qualification import QualificationRegime
from nerve.representation_optimizer.contracts import (
    canonical_json_bytes,
    stable_contract_id,
)
from nerve.representation_optimizer.lifecycle import OptimizationSession
from nerve.representation_optimizer.lifecycle import CandidateState


@dataclass(frozen=True)
class AutomatedOptimizationOutcome:
    report_path: Path
    report: Json
    output_package_dir: Path
    session: OptimizationSession


def build_structure_index(
    *,
    run_root: Path,
    analysis_directory: Path,
    evidence: Json,
) -> Json:
    claims = evidence["claims"]
    by_kind: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for claim in claims:
        kind = str(claim["kind"])
        status = str(claim["status"])
        by_kind[kind] = by_kind.get(kind, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1
    analyzer = dict(evidence["analyzer"])
    evidence_path = analysis_directory / "evidence" / f"{analyzer['id']}.json"
    return {
        "evidence_id": evidence["evidence_id"],
        "analyzer": analyzer,
        "evidence_ref": str(evidence_path.relative_to(run_root)),
        "claim_summary": {
            "total": len(claims),
            "exact": sum(bool(claim["exact"]) for claim in claims),
            "by_kind": dict(sorted(by_kind.items())),
            "by_status": dict(sorted(by_status.items())),
        },
    }


def build_report(
    *,
    run_id: str,
    package_id: str,
    source_package: Path,
    output_package: Path,
    status: str,
    budget: Json,
    budget_usage: Json,
    scopes: list[Json],
    provider_evaluations: list[Json],
    duplicate_candidates: list[Json],
    candidates: list[Json],
    promotions: list[Json],
    publication: Json,
    session: OptimizationSession,
    event_journal_ref: str,
    event_count: int,
) -> Json:
    document = {
        "schema": OPTIMIZER_REPORT_SCHEMA,
        "report_id": "",
        "run_id": run_id,
        "package_id": package_id,
        "status": status,
        "source_package": str(source_package),
        "output_package": str(output_package),
        "budget": deepcopy(budget),
        "budget_usage": deepcopy(budget_usage),
        "summary": _summary(
            scopes=scopes,
            provider_evaluations=provider_evaluations,
            duplicate_candidates=duplicate_candidates,
            candidates=candidates,
            promotions=promotions,
        ),
        "scopes": deepcopy(scopes),
        "provider_evaluations": deepcopy(provider_evaluations),
        "duplicate_candidates": deepcopy(duplicate_candidates),
        "candidates": deepcopy(candidates),
        "promotions": deepcopy(promotions),
        "publication": deepcopy(publication),
        "session": session.to_json(),
        "event_journal": {
            "artifact_ref": event_journal_ref,
            "event_count": event_count,
        },
    }
    unsigned = deepcopy(document)
    unsigned.pop("report_id")
    document["report_id"] = stable_contract_id("automated_optimizer_report", unsigned)
    validate_report(document)
    return document


def publish_report(run_root: Path, document: Json) -> Path:
    validate_report(document)
    path = run_root / "report.json"
    replace_json(path, document)
    validate_report_directory(run_root)
    return path


def validate_report(document: Json) -> None:
    fields = {
        "schema",
        "report_id",
        "run_id",
        "package_id",
        "status",
        "source_package",
        "output_package",
        "budget",
        "budget_usage",
        "summary",
        "scopes",
        "provider_evaluations",
        "duplicate_candidates",
        "candidates",
        "promotions",
        "publication",
        "session",
        "event_journal",
    }
    if not isinstance(document, dict) or set(document) != fields:
        raise ModelCompileError("automated optimizer report has invalid fields")
    if document["schema"] != OPTIMIZER_REPORT_SCHEMA:
        raise ModelCompileError("automated optimizer report has unsupported schema")
    if document["status"] not in {
        "completed",
        "completed_no_changes",
        "cancelled",
        "failed",
    }:
        raise ModelCompileError("automated optimizer report has invalid status")
    for field in (
        "budget",
        "budget_usage",
        "summary",
        "publication",
        "event_journal",
    ):
        if not isinstance(document[field], dict):
            raise ModelCompileError(
                f"automated optimizer report {field} must be an object"
            )
    for field in (
        "scopes",
        "provider_evaluations",
        "duplicate_candidates",
        "candidates",
        "promotions",
    ):
        if not isinstance(document[field], list):
            raise ModelCompileError(
                f"automated optimizer report {field} must be a list"
            )
    OptimizationSession.from_json(document["session"])
    _validate_scope_indexes(document["scopes"])
    _validate_report_semantics(document)
    unsigned = deepcopy(document)
    report_id = unsigned.pop("report_id")
    expected = stable_contract_id("automated_optimizer_report", unsigned)
    if report_id != expected:
        raise ModelCompileError(
            "automated optimizer report identity does not match its content"
        )
    canonical_json_bytes(document)


def validate_report_directory(run_root: Path) -> Json:
    report = read_object(run_root / "report.json")
    validate_report(report)
    run = read_object(run_root / "run.json")
    if (
        set(run)
        != {
            "schema",
            "run_id",
            "package_id",
            "source_package",
            "requested_output_package",
            "exact_baseline_digest",
            "target_ids",
            "qualification_regimes",
            "budget",
        }
        or run["schema"] != OPTIMIZER_RUN_SCHEMA
    ):
        raise ModelCompileError("automated optimizer run manifest is invalid")
    target_ids = run["target_ids"]
    qualification_regimes = run["qualification_regimes"]
    if (
        not isinstance(target_ids, list)
        or not isinstance(qualification_regimes, dict)
        or set(qualification_regimes) != set(target_ids)
    ):
        raise ModelCompileError(
            "automated optimizer qualification regimes are invalid"
        )
    for regime in qualification_regimes.values():
        if not isinstance(regime, dict):
            raise ModelCompileError(
                "automated optimizer qualification regime is invalid"
            )
        QualificationRegime.from_json(regime)
    if (
        run["run_id"] != report["run_id"]
        or run["package_id"] != report["package_id"]
        or run["source_package"] != report["source_package"]
        or run["budget"] != report["budget"]
        or not isinstance(run["target_ids"], list)
        or run["target_ids"] != sorted(set(run["target_ids"]))
        or not run["target_ids"]
    ):
        raise ModelCompileError(
            "automated optimizer report disagrees with its run manifest"
        )
    journal = report["event_journal"]
    ref = Path(journal["artifact_ref"])
    if ref.is_absolute() or ".." in ref.parts:
        raise ModelCompileError("optimizer event journal reference escapes run root")
    events = read_event_journal(run_root / ref)
    if len(events) != journal["event_count"]:
        raise ModelCompileError(
            "optimizer report event count does not match its journal"
        )
    if (
        len(events) < 2
        or events[0]["phase"] != "run"
        or events[0]["status"] != "started"
        or events[-1]["phase"] != "run"
        or events[-1]["status"] != report["status"]
        or sum(event["phase"] == "run" for event in events) != 2
    ):
        raise ModelCompileError(
            "optimizer event journal has no unique matching run boundary"
        )
    for event in events:
        for value in event["evidence_refs"]:
            evidence = _safe_run_ref(run_root, value, "event evidence")
            if not evidence.exists() and not evidence.is_symlink():
                raise ModelCompileError(f"optimizer event evidence is missing: {value}")
    for scope in report["scopes"]:
        if scope["analysis_ref"] is None:
            continue
        analysis_directory = _safe_run_ref(
            run_root,
            scope["analysis_ref"],
            "scope analysis",
        )
        if not analysis_directory.is_dir():
            raise ModelCompileError("optimizer scope analysis directory is missing")
        validate_analysis_run_directory(analysis_directory)
        for structure in scope["structures"]:
            evidence_path = _safe_run_ref(
                run_root,
                structure["evidence_ref"],
                "structure evidence",
            )
            if evidence_path.parent.parent != analysis_directory:
                raise ModelCompileError(
                    "optimizer structure evidence escapes its scope analysis"
                )
            evidence = read_object(evidence_path)
            expected = build_structure_index(
                run_root=run_root,
                analysis_directory=analysis_directory,
                evidence=evidence,
            )
            if structure != expected:
                raise ModelCompileError(
                    "optimizer structure index disagrees with canonical evidence"
                )
    source = Path(report["source_package"])
    requested_output = Path(run["requested_output_package"])
    output = Path(report["output_package"])
    if report["status"] == "completed":
        if output != requested_output or not output.is_dir():
            raise ModelCompileError(
                "completed optimizer report has no published output package"
            )
    elif output != source or requested_output.exists() or requested_output.is_symlink():
        raise ModelCompileError(
            "non-published optimizer run left an ambiguous output package"
        )
    return report


def _validate_report_semantics(document: Json) -> None:
    status = document["status"]
    source = document["source_package"]
    output = document["output_package"]
    publication_status = document["publication"].get("status")
    promotions = document["promotions"]
    expected_publication = {
        "completed": "published",
        "completed_no_changes": "not_required",
        "cancelled": "cancelled",
        "failed": "failed",
    }[status]
    if publication_status != expected_publication:
        raise ModelCompileError(
            "automated optimizer report status disagrees with publication"
        )
    if status == "completed":
        if not promotions or output == source:
            raise ModelCompileError(
                "completed optimizer report requires promoted output"
            )
    elif promotions and status == "completed_no_changes":
        raise ModelCompileError("no-change optimizer report cannot contain promotions")
    elif output != source:
        raise ModelCompileError(
            "unpublished optimizer report must retain the source package"
        )
    candidates = document["candidates"]
    candidate_ids = [candidate.get("candidate_id") for candidate in candidates]
    if (
        not all(isinstance(value, str) and value for value in candidate_ids)
        or candidate_ids != sorted(set(candidate_ids))
        or any(
            candidate.get("status") not in {state.value for state in CandidateState}
            for candidate in candidates
        )
    ):
        raise ModelCompileError(
            "automated optimizer candidate records are not canonical"
        )
    session = OptimizationSession.from_json(document["session"])
    session_by_id = {
        candidate.candidate_id: candidate for candidate in session.candidates
    }
    if any(
        session_by_id.get(candidate["candidate_id"]) is None
        or session_by_id[candidate["candidate_id"]].state.value != candidate["status"]
        for candidate in candidates
    ):
        raise ModelCompileError(
            "automated optimizer candidate records disagree with lifecycle session"
        )
    promotion_ids = [promotion.get("candidate_id") for promotion in promotions]
    if not all(value in set(candidate_ids) for value in promotion_ids) or len(
        promotion_ids
    ) != len(set(promotion_ids)):
        raise ModelCompileError(
            "automated optimizer promotions reference invalid candidates"
        )
    if status == "completed" and any(
        session_by_id[candidate_id].state != CandidateState.PUBLISHED
        for candidate_id in promotion_ids
    ):
        raise ModelCompileError(
            "published optimizer promotions lack published lifecycles"
        )
    expected_summary = _summary(
        scopes=document["scopes"],
        provider_evaluations=document["provider_evaluations"],
        duplicate_candidates=document["duplicate_candidates"],
        candidates=candidates,
        promotions=promotions,
    )
    if document["summary"] != expected_summary:
        raise ModelCompileError(
            "automated optimizer report summary disagrees with its records"
        )


def _validate_scope_indexes(scopes: list[Json]) -> None:
    scope_ids = []
    for scope in scopes:
        if not isinstance(scope, dict) or set(scope) != {
            "scope_id",
            "kind",
            "status",
            "reason",
            "analysis_ref",
            "structures",
        }:
            raise ModelCompileError(
                "automated optimizer scope record has invalid fields"
            )
        scope_id = scope["scope_id"]
        if not isinstance(scope_id, str) or not scope_id:
            raise ModelCompileError(
                "automated optimizer scope record has invalid identity"
            )
        scope_ids.append(scope_id)
        if scope["status"] not in {
            "analyzed",
            "budget_skipped",
            "provider_skipped",
            "cancelled",
            "failed",
        }:
            raise ModelCompileError(
                "automated optimizer scope record has invalid status"
            )
        if not isinstance(scope["reason"], str) or not scope["reason"]:
            raise ModelCompileError(
                "automated optimizer scope record requires a reason"
            )
        if not isinstance(scope["structures"], list):
            raise ModelCompileError(
                "automated optimizer scope structures must be a list"
            )
        if scope["status"] != "analyzed":
            if scope["analysis_ref"] is not None or scope["structures"]:
                raise ModelCompileError(
                    "unanalyzed optimizer scope cannot index evidence"
                )
            continue
        if not isinstance(scope["analysis_ref"], str):
            raise ModelCompileError(
                "analyzed optimizer scope requires an analysis reference"
            )
        evidence_ids = []
        analyzer_ids = []
        for structure in scope["structures"]:
            if not isinstance(structure, dict) or set(structure) != {
                "evidence_id",
                "analyzer",
                "evidence_ref",
                "claim_summary",
            }:
                raise ModelCompileError("optimizer structure index has invalid fields")
            evidence_ids.append(structure["evidence_id"])
            analyzer = structure["analyzer"]
            if (
                not isinstance(analyzer, dict)
                or set(analyzer) != {"id", "version"}
                or not all(
                    isinstance(value, str) and value for value in analyzer.values()
                )
            ):
                raise ModelCompileError(
                    "optimizer structure index has invalid analyzer identity"
                )
            analyzer_ids.append(analyzer["id"])
            if not isinstance(structure["evidence_ref"], str):
                raise ModelCompileError(
                    "optimizer structure index has invalid evidence reference"
                )
            summary = structure["claim_summary"]
            if not isinstance(summary, dict) or set(summary) != {
                "total",
                "exact",
                "by_kind",
                "by_status",
            }:
                raise ModelCompileError(
                    "optimizer structure index has invalid claim summary"
                )
            if (
                not isinstance(summary["total"], int)
                or isinstance(summary["total"], bool)
                or summary["total"] <= 0
                or not isinstance(summary["exact"], int)
                or isinstance(summary["exact"], bool)
                or not 0 <= summary["exact"] <= summary["total"]
            ):
                raise ModelCompileError(
                    "optimizer structure index has invalid claim counts"
                )
            for field in ("by_kind", "by_status"):
                counts = summary[field]
                if (
                    not isinstance(counts, dict)
                    or not counts
                    or any(
                        not isinstance(key, str)
                        or not key
                        or not isinstance(value, int)
                        or isinstance(value, bool)
                        or value <= 0
                        for key, value in counts.items()
                    )
                    or sum(counts.values()) != summary["total"]
                ):
                    raise ModelCompileError(
                        "optimizer structure index has inconsistent claim counts"
                    )
            if set(summary["by_status"]) - {
                "supported",
                "rejected",
                "inconclusive",
            }:
                raise ModelCompileError(
                    "optimizer structure index has invalid claim status"
                )
        if len(evidence_ids) != len(set(evidence_ids)) or analyzer_ids != sorted(
            set(analyzer_ids)
        ):
            raise ModelCompileError(
                "optimizer structure indexes must be sorted and unique"
            )
    if scope_ids != sorted(set(scope_ids)):
        raise ModelCompileError(
            "automated optimizer scope records must be sorted and unique"
        )


def _safe_run_ref(run_root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ModelCompileError(f"optimizer {label} is invalid")
    relative = Path(value)
    if relative.is_absolute() or "." in relative.parts or ".." in relative.parts:
        raise ModelCompileError(f"optimizer {label} escapes run root")
    return run_root / relative


def _summary(
    *,
    scopes: list[Json],
    provider_evaluations: list[Json],
    duplicate_candidates: list[Json],
    candidates: list[Json],
    promotions: list[Json],
) -> Json:
    statuses: dict[str, int] = {}
    for candidate in candidates:
        status = str(candidate["status"])
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "scope_count": len(scopes),
        "analyzed_scope_count": sum(scope["status"] == "analyzed" for scope in scopes),
        "analysis_failure_count": sum(scope["status"] == "failed" for scope in scopes),
        "provider_evaluation_count": len(provider_evaluations),
        "provider_failure_count": sum(
            item["status"] == "failed" for item in provider_evaluations
        ),
        "candidate_count": len(candidates),
        "candidate_status_counts": dict(sorted(statuses.items())),
        "deduplicated_candidate_count": len(duplicate_candidates),
        "materially_faster_count": sum(
            candidate.get("benchmark_decision") == "materially_faster"
            for candidate in candidates
        ),
        "faster_but_invalid_count": sum(
            candidate.get("benchmark_decision") == "materially_faster"
            and candidate.get("validation_status") not in {None, "passed"}
            for candidate in candidates
        ),
        "promotion_count": len(promotions),
    }
