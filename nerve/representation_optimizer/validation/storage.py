from __future__ import annotations

import json
import os
import shutil
from hashlib import sha256
from pathlib import Path, PurePosixPath
from uuid import uuid4

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import (
    BENCHMARK_RECORD_SCHEMA,
    VALIDATION_RECORD_SCHEMA,
    ContractDocument,
    canonical_json_bytes,
)
from nerve.representation_optimizer.validation.contracts import (
    PREBENCHMARK_RECORD_SCHEMA,
    VALIDATION_EVIDENCE_INTEGRITY_SCHEMA,
    ValidationPlan,
    ValidationRun,
)
from nerve.representation_optimizer.validation.protocols import (
    BehavioralValidationAdapter,
)
from nerve.representation_optimizer.validation.proofs import (
    ProofVerifierRegistry,
)


VALIDATION_INTEGRITY_FILE = "integrity.json"


def publish_prebenchmark_evidence(
    workspace_root: Path,
    *,
    plan: ValidationPlan,
    record: ContractDocument,
    sanity_run: ValidationRun | None,
    artifact_source: BehavioralValidationAdapter,
    proof_artifact_source: ProofVerifierRegistry,
) -> Path:
    if record.schema != PREBENCHMARK_RECORD_SCHEMA:
        raise ModelCompileError(
            "prebenchmark publication requires a prebenchmark record"
        )
    record_document = record.to_json()
    evidence_id = str(record_document["prebenchmark_id"])
    documents: dict[str, Json] = {
        "plan.json": plan.to_json(),
        "record.json": record_document,
    }
    if sanity_run is not None:
        documents["sanity_run.json"] = sanity_run.to_json()
    return _publish_evidence(
        workspace_root=workspace_root,
        collection="prebenchmark",
        evidence_id=evidence_id,
        documents=documents,
        fixture_refs=_fixture_refs(
            plan,
            stages=("sanity",),
        ),
        trace_refs=(
            _trace_refs((sanity_run,))
            if sanity_run is not None
            else ()
        ),
        artifact_source=artifact_source,
        candidate_id=plan.candidate_id,
        extra_artifacts=_proof_artifact_readers(
            record_document["proof_results"],
            proof_artifact_source,
        ),
    )


def publish_validation_evidence(
    workspace_root: Path,
    *,
    plan: ValidationPlan,
    prebenchmark_record: ContractDocument,
    benchmark_record: ContractDocument,
    runs: tuple[ValidationRun, ...],
    record: ContractDocument,
    artifact_source: BehavioralValidationAdapter,
) -> Path:
    if (
        prebenchmark_record.schema != PREBENCHMARK_RECORD_SCHEMA
        or benchmark_record.schema != BENCHMARK_RECORD_SCHEMA
        or record.schema != VALIDATION_RECORD_SCHEMA
    ):
        raise ModelCompileError(
            "validation publication received incompatible evidence schemas"
        )
    record_document = record.to_json()
    evidence_id = str(record_document["validation_id"])
    documents: dict[str, Json] = {
        "plan.json": plan.to_json(),
        "prebenchmark_record.json": prebenchmark_record.to_json(),
        "benchmark_record.json": benchmark_record.to_json(),
        "record.json": record_document,
    }
    for run in runs:
        run_document = run.to_json()
        documents[f"{run_document['stage']}_run.json"] = run_document
    return _publish_evidence(
        workspace_root=workspace_root,
        collection="validations",
        evidence_id=evidence_id,
        documents=documents,
        fixture_refs=_fixture_refs(
            plan,
            stages=("full_local", "whole_model"),
        ),
        trace_refs=_trace_refs(runs),
        artifact_source=artifact_source,
        candidate_id=plan.candidate_id,
        extra_artifacts=(),
    )


def load_prebenchmark_evidence(
    workspace_root: Path,
    prebenchmark_id: str,
) -> tuple[ValidationPlan, ContractDocument, ValidationRun | None]:
    root = _evidence_root(
        workspace_root,
        "prebenchmark",
        prebenchmark_id,
        "prebenchmark_validation",
    )
    _validate_evidence_tree(root, prebenchmark_id)
    plan = ValidationPlan.from_json(_read_object(root / "plan.json"))
    record = ContractDocument.from_json(
        _read_object(root / "record.json"),
        expected_schema=PREBENCHMARK_RECORD_SCHEMA,
    )
    run_path = root / "sanity_run.json"
    run = (
        ValidationRun.from_json(_read_object(run_path))
        if run_path.exists()
        else None
    )
    _validate_prebenchmark_links(plan, record, run)
    _validate_expected_artifacts(
        root,
        (
            *_fixture_refs(plan, stages=("sanity",)),
            *(_trace_refs((run,)) if run is not None else ()),
            *(
                reference
                for result in record.to_json()["proof_results"]
                for reference in result["artifacts"]
            ),
        ),
    )
    return plan, record, run


def load_validation_evidence(
    workspace_root: Path,
    validation_id: str,
) -> tuple[
    ValidationPlan,
    ContractDocument,
    ContractDocument,
    tuple[ValidationRun, ...],
    ContractDocument,
]:
    root = _evidence_root(
        workspace_root,
        "validations",
        validation_id,
        "validation",
    )
    _validate_evidence_tree(root, validation_id)
    plan = ValidationPlan.from_json(_read_object(root / "plan.json"))
    prebenchmark = ContractDocument.from_json(
        _read_object(root / "prebenchmark_record.json"),
        expected_schema=PREBENCHMARK_RECORD_SCHEMA,
    )
    benchmark = ContractDocument.from_json(
        _read_object(root / "benchmark_record.json"),
        expected_schema=BENCHMARK_RECORD_SCHEMA,
    )
    runs = tuple(
        ValidationRun.from_json(_read_object(root / f"{stage}_run.json"))
        for stage in ("full_local", "whole_model")
        if (root / f"{stage}_run.json").exists()
    )
    record = ContractDocument.from_json(
        _read_object(root / "record.json"),
        expected_schema=VALIDATION_RECORD_SCHEMA,
    )
    _validate_validation_links(
        plan,
        prebenchmark,
        benchmark,
        runs,
        record,
    )
    _validate_expected_artifacts(
        root,
        (
            *_fixture_refs(
                plan,
                stages=("full_local", "whole_model"),
            ),
            *_trace_refs(runs),
        ),
    )
    return plan, prebenchmark, benchmark, runs, record


def _publish_evidence(
    *,
    workspace_root: Path,
    collection: str,
    evidence_id: str,
    documents: dict[str, Json],
    fixture_refs: tuple[Json, ...],
    trace_refs: tuple[Json, ...],
    artifact_source: BehavioralValidationAdapter,
    candidate_id: str,
    extra_artifacts: tuple[tuple[Json, object], ...],
) -> Path:
    workspace_root = workspace_root.resolve()
    collection_root = workspace_root / collection
    if collection_root.is_symlink():
        raise ModelCompileError(
            "validation evidence collection must not be a symlink"
        )
    ready = collection_root / evidence_id
    if ready.is_symlink():
        raise ModelCompileError(
            "validation evidence path must not be a symlink"
        )
    if ready.exists():
        _validate_evidence_tree(ready, evidence_id)
        for relative_path, document in documents.items():
            if _read_object(ready / relative_path) != document:
                raise ModelCompileError(
                    "validation evidence identity is bound to "
                    "different documents"
                )
        _validate_expected_artifacts(
            ready,
            (
                *fixture_refs,
                *trace_refs,
                *(reference for reference, _reader in extra_artifacts),
            ),
        )
        return ready
    staging_root = workspace_root / ".validation-staging"
    if staging_root.is_symlink():
        raise ModelCompileError(
            "validation evidence staging must not be a symlink"
        )
    staging = staging_root / f"{evidence_id}.{uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    _fsync_directory(staging_root)
    _fsync_directory(workspace_root)
    published = False
    try:
        for relative_path, document in sorted(documents.items()):
            _write_json(staging / relative_path, document)
        _copy_artifacts(
            staging,
            fixture_refs,
            lambda relative_path: artifact_source.iter_fixture_artifact(
                relative_path,
                candidate_id=candidate_id,
            ),
        )
        _copy_artifacts(
            staging,
            trace_refs,
            artifact_source.iter_trace_artifact,
        )
        for reference, reader in extra_artifacts:
            _copy_artifacts(staging, (reference,), reader)
        _write_json(
            staging / VALIDATION_INTEGRITY_FILE,
            _integrity_document(staging, evidence_id),
        )
        _validate_evidence_tree(staging, evidence_id)
        _fsync_tree_directories(staging)
        ready.parent.mkdir(parents=True, exist_ok=True)
        _fsync_directory(workspace_root)
        staging.replace(ready)
        published = True
        _fsync_directory(staging_root)
        _fsync_directory(ready.parent)
        _validate_evidence_tree(ready, evidence_id)
        _validate_expected_artifacts(
            ready,
            (
                *fixture_refs,
                *trace_refs,
                *(reference for reference, _reader in extra_artifacts),
            ),
        )
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        if published and ready.exists():
            shutil.rmtree(ready)
        if staging.exists() or (published and ready.exists()):
            raise ModelCompileError(
                "failed to clean incomplete validation evidence"
            )
        raise
    return ready


def _validate_expected_artifacts(
    root: Path,
    references: tuple[Json, ...],
) -> None:
    for reference in references:
        relative = _safe_relative(reference["path"])
        path = root / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or (
                "nerve.optimizer.artifact_sha256.v1:"
                f"{_file_sha256(path)}"
            )
            != reference["digest"]
        ):
            raise ModelCompileError(
                f"validation evidence is missing a declared artifact: "
                f"{relative}"
            )


def _proof_artifact_readers(
    proof_results: list[Json],
    source: ProofVerifierRegistry,
) -> tuple[tuple[Json, object], ...]:
    artifacts = []
    paths: set[str] = set()
    for result in proof_results:
        verifier_id = str(result["verifier_id"])
        for reference in result["artifacts"]:
            path = str(reference["path"])
            if path in paths:
                raise ModelCompileError(
                    "proof results reuse an artifact path"
                )
            paths.add(path)

            def reader(
                relative_path,
                *,
                chunk_bytes=8 * 1024 * 1024,
                selected_verifier=verifier_id,
            ):
                yield from source.iter_proof_artifact(
                    selected_verifier,
                    relative_path,
                    chunk_bytes=chunk_bytes,
                )

            artifacts.append((dict(reference), reader))
    return tuple(artifacts)


def _fixture_refs(
    plan: ValidationPlan,
    *,
    stages: tuple[str, ...],
) -> tuple[Json, ...]:
    by_path: dict[str, Json] = {}
    for stage in stages:
        for check in plan.checks_for_stage(stage):
            for field in ("input", "initial_state"):
                reference = check[field]
                if reference is None:
                    continue
                previous = by_path.setdefault(
                    str(reference["path"]),
                    dict(reference),
                )
                if previous != reference:
                    raise ModelCompileError(
                        "validation fixture path has conflicting digests"
                    )
            for basis in (
                check["regime"]["context_size_basis"],
                check["horizon"]["output_allowance_basis"],
            ):
                if basis["kind"] != "declared_model_limit":
                    continue
                reference = basis["artifact"]
                previous = by_path.setdefault(
                    str(reference["path"]),
                    dict(reference),
                )
                if previous != reference:
                    raise ModelCompileError(
                        "validation limit evidence path has conflicting digests"
                    )
    return tuple(by_path[path] for path in sorted(by_path))


def _trace_refs(
    runs: tuple[ValidationRun | None, ...],
) -> tuple[Json, ...]:
    by_path: dict[str, Json] = {}
    for run in runs:
        if run is None:
            continue
        for observation in run.to_json()["observations"]:
            for role in ("reference", "candidate"):
                for reference in observation["traces"][role]:
                    path = str(reference["path"])
                    previous = by_path.setdefault(path, dict(reference))
                    if previous != reference:
                        raise ModelCompileError(
                            "validation trace path has conflicting digests"
                        )
    return tuple(by_path[path] for path in sorted(by_path))


def _copy_artifacts(
    root: Path,
    references: tuple[Json, ...],
    reader,
) -> None:
    for reference in references:
        relative_path = _safe_relative(str(reference["path"]))
        target = root / relative_path
        if target.exists() or target.is_symlink():
            raise ModelCompileError(
                f"validation evidence artifact collision: {relative_path}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        digest = sha256()
        chunks = 0
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                for chunk in reader(relative_path.as_posix()):
                    if not isinstance(chunk, bytes) or not chunk:
                        raise ModelCompileError(
                            "validation evidence readers must yield "
                            "non-empty bytes"
                        )
                    stream.write(chunk)
                    digest.update(chunk)
                    chunks += 1
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(descriptor)
        observed = (
            "nerve.optimizer.artifact_sha256.v1:"
            f"{digest.hexdigest()}"
        )
        if chunks == 0 or observed != reference["digest"]:
            raise ModelCompileError(
                f"validation evidence artifact failed integrity: "
                f"{relative_path}"
            )


def _integrity_document(root: Path, evidence_id: str) -> Json:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ModelCompileError(
                "validation evidence contains a symbolic link"
            )
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "byte_count": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    return {
        "schema": VALIDATION_EVIDENCE_INTEGRITY_SCHEMA,
        "evidence_id": evidence_id,
        "files": files,
    }


def _validate_evidence_tree(root: Path, evidence_id: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ModelCompileError(
            "validation evidence directory is missing or unsafe"
        )
    integrity = _read_object(root / VALIDATION_INTEGRITY_FILE)
    if (
        set(integrity)
        != {"schema", "evidence_id", "files"}
        or integrity["schema"] != VALIDATION_EVIDENCE_INTEGRITY_SCHEMA
        or integrity["evidence_id"] != evidence_id
        or not isinstance(integrity["files"], list)
    ):
        raise ModelCompileError(
            "validation evidence integrity manifest is invalid"
        )
    paths: list[str] = []
    for record in integrity["files"]:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "byte_count", "sha256"}
        ):
            raise ModelCompileError(
                "validation evidence file record is invalid"
            )
        relative = _safe_relative(record["path"])
        path = root / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record["byte_count"]
            or _file_sha256(path) != record["sha256"]
        ):
            raise ModelCompileError(
                f"validation evidence failed integrity: {relative}"
            )
        paths.append(relative.as_posix())
    if paths != sorted(set(paths)):
        raise ModelCompileError(
            "validation evidence integrity paths are not canonical"
        )
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != VALIDATION_INTEGRITY_FILE
    }
    if set(paths) != actual or any(
        path.is_symlink() for path in root.rglob("*")
    ):
        raise ModelCompileError(
            "validation evidence integrity manifest is incomplete"
        )


def _validate_prebenchmark_links(
    plan: ValidationPlan,
    record: ContractDocument,
    run: ValidationRun | None,
) -> None:
    document = record.to_json()
    if (
        document["candidate_id"] != plan.candidate_id
        or document["validation_plan_digest"]
        != _contract_digest(plan.to_json())
        or document["sanity_run_digest"]
        != (None if run is None else _contract_digest(run.to_json()))
    ):
        raise ModelCompileError(
            "prebenchmark validation evidence does not match"
        )


def _validate_validation_links(
    plan: ValidationPlan,
    prebenchmark: ContractDocument,
    benchmark: ContractDocument,
    runs: tuple[ValidationRun, ...],
    record: ContractDocument,
) -> None:
    document = record.to_json()
    if (
        document["candidate_id"] != plan.candidate_id
        or document["validation_plan_digest"]
        != _contract_digest(plan.to_json())
        or document["prebenchmark_record_digest"]
        != prebenchmark.digest
        or document["benchmark_record_digest"] != benchmark.digest
        or document["runs"]
        != [
            {
                "stage": run.to_json()["stage"],
                "run_digest": _contract_digest(run.to_json()),
            }
            for run in runs
        ]
    ):
        raise ModelCompileError(
            "final validation evidence does not match"
        )


def _contract_digest(document: Json) -> str:
    from nerve.representation_optimizer.contracts import contract_digest

    return contract_digest(document)


def _evidence_root(
    workspace_root: Path,
    collection: str,
    evidence_id: str,
    prefix: str,
) -> Path:
    if (
        not evidence_id.startswith(f"{prefix}_")
        or len(evidence_id) != len(prefix) + 33
        or any(
            character not in "0123456789abcdef"
            for character in evidence_id.rsplit("_", 1)[-1]
        )
    ):
        raise ModelCompileError(
            "validation evidence identity is invalid"
        )
    collection_root = workspace_root.resolve() / collection
    if collection_root.is_symlink():
        raise ModelCompileError(
            "validation evidence collection must not be a symlink"
        )
    root = collection_root / evidence_id
    if root.is_symlink():
        raise ModelCompileError(
            "validation evidence path must not be a symlink"
        )
    return root


def _safe_relative(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ModelCompileError(
            "validation evidence path must be a non-empty string"
        )
    posix = PurePosixPath(value)
    if (
        posix.is_absolute()
        or "." in posix.parts
        or ".." in posix.parts
        or posix.as_posix() != value
    ):
        raise ModelCompileError(
            f"validation evidence path is unsafe: {value!r}"
        )
    return Path(*posix.parts)


def _write_json(path: Path, document: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(document) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _read_object(path: Path) -> Json:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ModelCompileError(
            f"validation evidence document is unreadable: {path}"
        ) from error
    if not isinstance(document, dict):
        raise ModelCompileError(
            f"validation evidence document must be an object: {path}"
        )
    return document


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_tree_directories(root: Path) -> None:
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        _fsync_directory(directory)
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
