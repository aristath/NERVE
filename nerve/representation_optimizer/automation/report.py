from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.automation.contracts import (
    OPTIMIZER_REPORT_SCHEMA,
    OPTIMIZER_RUN_SCHEMA,
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
from nerve.representation_optimizer.lifecycle import CandidateState


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
    if set(run) != {
        "schema",
        "run_id",
        "package_id",
        "source_package",
        "requested_output_package",
        "exact_baseline_digest",
        "target_ids",
        "budget",
    } or run["schema"] != OPTIMIZER_RUN_SCHEMA:
        raise ModelCompileError("automated optimizer run manifest is invalid")
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
                raise ModelCompileError(
                    f"optimizer event evidence is missing: {value}"
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
        raise ModelCompileError(
            "no-change optimizer report cannot contain promotions"
        )
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
            candidate.get("status")
            not in {state.value for state in CandidateState}
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
        or session_by_id[candidate["candidate_id"]].state.value
        != candidate["status"]
        for candidate in candidates
    ):
        raise ModelCompileError(
            "automated optimizer candidate records disagree with lifecycle session"
        )
    promotion_ids = [promotion.get("candidate_id") for promotion in promotions]
    if (
        not all(value in set(candidate_ids) for value in promotion_ids)
        or len(promotion_ids) != len(set(promotion_ids))
    ):
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
