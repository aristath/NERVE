from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.automation.contracts import (
    OPTIMIZER_REPORT_SCHEMA,
)
from nerve.representation_optimizer.automation.events import read_event_journal
from nerve.representation_optimizer.automation.storage import (
    read_object,
    replace_json,
)
from nerve.representation_optimizer.contracts import (
    canonical_json_bytes,
    stable_contract_id,
)
from nerve.representation_optimizer.lifecycle import OptimizationSession


@dataclass(frozen=True)
class AutomatedOptimizationOutcome:
    report_path: Path
    report: Json
    output_package_dir: Path
    session: OptimizationSession


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
    if document["status"] not in {"completed", "completed_no_changes", "failed"}:
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
    journal = report["event_journal"]
    ref = Path(journal["artifact_ref"])
    if ref.is_absolute() or ".." in ref.parts:
        raise ModelCompileError("optimizer event journal reference escapes run root")
    events = read_event_journal(run_root / ref)
    if len(events) != journal["event_count"]:
        raise ModelCompileError(
            "optimizer report event count does not match its journal"
        )
    return report


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
        "analyzed_scope_count": sum(
            scope["status"] == "analyzed" for scope in scopes
        ),
        "analysis_failure_count": sum(
            scope["status"] == "failed" for scope in scopes
        ),
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
