from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from nerve.compilation import Json, ModelCompileError, read_json, write_json
from nerve.representation_optimizer.contracts import (
    ALGEBRAIC_EVIDENCE_SCHEMA,
    algebraic_evidence_id,
    contract_digest,
    stable_contract_id,
    validate_contract,
)


ANALYSIS_RUN_SCHEMA = "nerve.optimizer.analysis_run.v1"


@dataclass(frozen=True)
class AnalysisRun:
    document: Json
    evidence: tuple[Json, ...]
    details: tuple[Json, ...]

    @property
    def run_id(self) -> str:
        return str(self.document["run_id"])


def build_evidence(
    *,
    scope_id: str,
    source_contract_digest: str,
    analyzer_id: str,
    analyzer_version: str,
    claims: tuple[Json, ...],
    details: Json,
) -> tuple[Json, Json]:
    details_document = {
        "schema": "nerve.optimizer.analysis_details.v1",
        "scope_id": scope_id,
        "analyzer": {"id": analyzer_id, "version": analyzer_version},
        "details": details,
    }
    _validate_details_document(
        details_document,
        scope_id=scope_id,
        analyzer={"id": analyzer_id, "version": analyzer_version},
    )
    details_digest = contract_digest(details_document)
    evidence: Json = {
        "schema": ALGEBRAIC_EVIDENCE_SCHEMA,
        "evidence_id": "",
        "scope_id": scope_id,
        "source_contract_digest": source_contract_digest,
        "analyzer": {"id": analyzer_id, "version": analyzer_version},
        "claims": list(claims),
        "artifacts": [
            {
                "path": f"details/{analyzer_id}.json",
                "digest": details_digest,
            }
        ],
    }
    evidence["evidence_id"] = algebraic_evidence_id(evidence)
    validate_contract(evidence, expected_schema=ALGEBRAIC_EVIDENCE_SCHEMA)
    return evidence, details_document


def build_analysis_run(
    *,
    package_id: str,
    scope_id: str,
    source_contract_digest: str,
    budget: Json,
    evidence: tuple[Json, ...],
    details: tuple[Json, ...],
) -> AnalysisRun:
    if len(evidence) != len(details):
        raise ModelCompileError("analysis evidence and details do not align")
    evidence = tuple(sorted(evidence, key=lambda item: str(item["analyzer"]["id"])))
    by_analyzer = {str(item["analyzer"]["id"]): item for item in details}
    details = tuple(by_analyzer[str(item["analyzer"]["id"])] for item in evidence)
    document: Json = {
        "schema": ANALYSIS_RUN_SCHEMA,
        "run_id": "",
        "package_id": package_id,
        "scope_id": scope_id,
        "source_contract_digest": source_contract_digest,
        "budget": budget,
        "evidence": [
            {
                "evidence_id": item["evidence_id"],
                "analyzer": item["analyzer"],
                "path": f"evidence/{item['analyzer']['id']}.json",
                "digest": contract_digest(item),
                "details_path": item["artifacts"][0]["path"],
                "details_digest": item["artifacts"][0]["digest"],
            }
            for item in evidence
        ],
    }
    unsigned = deepcopy(document)
    unsigned.pop("run_id")
    document["run_id"] = stable_contract_id("analysis_run", unsigned)
    validate_analysis_run(document)
    return AnalysisRun(document=document, evidence=evidence, details=details)


def write_analysis_run(run: AnalysisRun, output_dir: Path) -> Path:
    if output_dir.exists():
        raise ModelCompileError(
            f"analysis output already exists; refusing to mutate it: {output_dir}"
        )
    temporary = output_dir.with_name(f".{output_dir.name}.{run.run_id}.tmp")
    if temporary.exists():
        raise ModelCompileError(
            f"stale analysis output staging directory exists: {temporary}"
        )
    try:
        for evidence, details in zip(
            run.evidence,
            run.details,
            strict=True,
        ):
            analyzer_id = str(evidence["analyzer"]["id"])
            write_json(temporary / "evidence" / f"{analyzer_id}.json", evidence)
            write_json(temporary / "details" / f"{analyzer_id}.json", details)
        write_json(temporary / "analysis.json", run.document)
        validate_analysis_run_directory(temporary)
        temporary.rename(output_dir)
    except BaseException:
        _remove_incomplete_tree(temporary)
        raise
    return output_dir / "analysis.json"


def validate_analysis_run(document: Json) -> None:
    required = {
        "schema",
        "run_id",
        "package_id",
        "scope_id",
        "source_contract_digest",
        "budget",
        "evidence",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise ModelCompileError("analysis run has invalid fields")
    if document["schema"] != ANALYSIS_RUN_SCHEMA:
        raise ModelCompileError("analysis run has unsupported schema")
    for field in ("package_id", "scope_id", "source_contract_digest"):
        if not isinstance(document[field], str) or not document[field]:
            raise ModelCompileError(f"analysis run {field} must be a string")
    if not isinstance(document["budget"], dict):
        raise ModelCompileError("analysis run budget must be an object")
    records = document["evidence"]
    if not isinstance(records, list) or not records:
        raise ModelCompileError("analysis run must contain evidence")
    analyzers = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "evidence_id",
            "analyzer",
            "path",
            "digest",
            "details_path",
            "details_digest",
        }:
            raise ModelCompileError("analysis run evidence reference is invalid")
        analyzer = record["analyzer"]
        if (
            not isinstance(analyzer, dict)
            or set(analyzer) != {"id", "version"}
            or not all(
                isinstance(analyzer[field], str) and analyzer[field]
                for field in ("id", "version")
            )
        ):
            raise ModelCompileError("analysis run analyzer identity is invalid")
        analyzers.append(analyzer["id"])
        for field in ("path", "details_path"):
            _safe_relative_path(record[field], f"analysis run {field}")
        for field in ("digest", "details_digest"):
            if not isinstance(record[field], str) or ":" not in record[field]:
                raise ModelCompileError(f"analysis run {field} is invalid")
    if analyzers != sorted(set(analyzers)):
        raise ModelCompileError("analysis run analyzers must be sorted and unique")
    unsigned = deepcopy(document)
    run_id = unsigned.pop("run_id")
    expected = stable_contract_id("analysis_run", unsigned)
    if run_id != expected:
        raise ModelCompileError(
            "analysis run identity does not match canonical content"
        )


def validate_analysis_run_directory(output_dir: Path) -> AnalysisRun:
    document = read_json(output_dir / "analysis.json")
    validate_analysis_run(document)
    evidence = []
    details = []
    for record in document["evidence"]:
        evidence_path = output_dir / _safe_relative_path(
            record["path"],
            "analysis evidence path",
        )
        detail_path = output_dir / _safe_relative_path(
            record["details_path"],
            "analysis details path",
        )
        evidence_document = read_json(evidence_path)
        details_document = read_json(detail_path)
        validate_contract(
            evidence_document,
            expected_schema=ALGEBRAIC_EVIDENCE_SCHEMA,
        )
        if evidence_document["evidence_id"] != record["evidence_id"]:
            raise ModelCompileError("analysis evidence identity does not match index")
        if contract_digest(evidence_document) != record["digest"]:
            raise ModelCompileError("analysis evidence digest does not match index")
        if contract_digest(details_document) != record["details_digest"]:
            raise ModelCompileError("analysis details digest does not match index")
        _validate_details_document(
            details_document,
            scope_id=document["scope_id"],
            analyzer=record["analyzer"],
        )
        evidence.append(evidence_document)
        details.append(details_document)
    return AnalysisRun(
        document=document,
        evidence=tuple(evidence),
        details=tuple(details),
    )


def _validate_details_document(
    document: Json,
    *,
    scope_id: str,
    analyzer: Json,
) -> None:
    if (
        not isinstance(document, dict)
        or set(document) != {"schema", "scope_id", "analyzer", "details"}
        or document["schema"] != "nerve.optimizer.analysis_details.v1"
        or document["scope_id"] != scope_id
        or document["analyzer"] != analyzer
        or not isinstance(document["details"], dict)
    ):
        raise ModelCompileError("analysis details artifact is invalid")


def _safe_relative_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ModelCompileError(f"{label} must be a non-empty path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ModelCompileError(f"{label} must remain inside analysis output")
    return path


def _remove_incomplete_tree(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()
