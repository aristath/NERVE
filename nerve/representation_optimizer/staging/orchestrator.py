from __future__ import annotations

import fcntl
import json
import os
import resource
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator
from uuid import uuid4

from nerve.compilation import Json, ModelCompileCancelled, ModelCompileError
from nerve.representation_optimizer.contracts import (
    CANDIDATE_CONSTRUCTION_SCHEMA,
    RELOWERING_REQUEST_SCHEMA,
    ContractDocument,
    contract_digest,
    stable_contract_id,
)
from nerve.representation_optimizer.lifecycle import (
    CandidateState,
    OptimizationSession,
)
from nerve.representation_optimizer.providers.types import ProviderCandidatePlan
from nerve.representation_optimizer.staging.artifact_validation import (
    ArtifactValidatorRegistry,
)
from nerve.representation_optimizer.staging.contracts import (
    CONSTRUCTION_PHASES,
    CandidateBuildPlan,
)
from nerve.representation_optimizer.staging.integrity import (
    integrity_evidence,
    validate_staged_candidate,
    write_staged_candidate_integrity,
)
from nerve.representation_optimizer.staging.loading import (
    load_staged_candidate,
)
from nerve.representation_optimizer.staging.protocols import (
    CandidateOrdinaryRelowerer,
    CandidatePhysicalOptimizer,
    CandidateSemanticConstructor,
)
from nerve.representation_optimizer.staging.source_seal import (
    seal_source_package,
    verify_source_package_seal,
)
from nerve.representation_optimizer.staging.workspace import (
    CandidateConstructionContext,
)


@dataclass(frozen=True)
class CandidateConstructionOutcome:
    status: str
    record: ContractDocument
    session: OptimizationSession
    staged_candidate_path: Path | None


@contextmanager
def _candidate_lock(
    workspace_root: Path,
    candidate_id: str,
) -> Iterator[None]:
    lock_root = workspace_root / "locks"
    if lock_root.is_symlink():
        raise ModelCompileError("candidate lock directory must not be a symlink")
    lock_root.mkdir(parents=True, exist_ok=True)
    path = lock_root / f"{candidate_id}.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as error:
        raise ModelCompileError(
            f"candidate construction lock is unavailable: {candidate_id!r}"
        ) from error
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ModelCompileError(
                f"candidate is already being constructed: {candidate_id!r}"
            ) from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def stage_candidate(
    *,
    package_dir: Path,
    workspace_root: Path,
    plan: ProviderCandidatePlan,
    session: OptimizationSession,
    semantic_constructor: CandidateSemanticConstructor,
    ordinary_relowerer: CandidateOrdinaryRelowerer,
    physical_optimizer: CandidatePhysicalOptimizer,
    artifact_validators: ArtifactValidatorRegistry | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> CandidateConstructionOutcome:
    package_dir = package_dir.resolve()
    workspace_root = workspace_root.resolve()
    _validate_isolation(package_dir, workspace_root)
    candidate = plan.candidate.to_json()
    candidate_id = plan.candidate_id
    build_plan = plan.construction_requirements
    _validate_candidate_build_binding(plan)
    _validate_session(session, candidate, package_dir)
    started_ns = time.monotonic_ns()
    initial_rss = _resident_bytes()
    source_seal = seal_source_package(package_dir, build_plan)

    with _candidate_lock(workspace_root, candidate_id):
        recovered = _recover_published_candidate(
            package_dir=package_dir,
            workspace_root=workspace_root,
            plan=plan,
            session=session,
        )
        if recovered is not None:
            return recovered
        _remove_abandoned_staging(workspace_root, candidate_id)
        return _construct_candidate(
            package_dir=package_dir,
            workspace_root=workspace_root,
            plan=plan,
            session=session,
            candidate=candidate,
            build_plan=build_plan,
            source_seal=source_seal,
            started_ns=started_ns,
            initial_rss=initial_rss,
            semantic_constructor=semantic_constructor,
            ordinary_relowerer=ordinary_relowerer,
            physical_optimizer=physical_optimizer,
            artifact_validators=artifact_validators,
            cancel_requested=cancel_requested,
        )


def _recover_published_candidate(
    *,
    package_dir: Path,
    workspace_root: Path,
    plan: ProviderCandidatePlan,
    session: OptimizationSession,
) -> CandidateConstructionOutcome | None:
    ready_path = workspace_root / "ready" / plan.candidate_id
    if ready_path.is_symlink():
        raise ModelCompileError("published candidate must not be a symlink")
    if not ready_path.exists():
        return None
    integrity = validate_staged_candidate(
        ready_path,
        expected_candidate_id=plan.candidate_id,
        expected_build_plan=plan.construction_requirements,
    )
    record_path = (
        workspace_root
        / "records"
        / f"{integrity['construction_id']}.json"
    )
    if record_path.is_symlink():
        raise ModelCompileError(
            "published candidate construction record must not be a symlink"
        )
    if not record_path.exists():
        _remove_candidate_tree(ready_path)
        return None
    loaded = load_staged_candidate(
        workspace_root,
        plan.candidate_id,
        package_dir=package_dir,
    )
    if loaded.build_plan != plan.construction_requirements:
        raise ModelCompileError(
            "published candidate build plan does not match requested plan"
        )
    if _read_object(loaded.path / "contracts" / "candidate.json") != (
        plan.candidate.to_json()
    ):
        raise ModelCompileError(
            "published candidate contract does not match requested plan"
        )
    record_document = loaded.record.to_json()
    expected_digests = {
        "representation_graph_digest": contract_digest(
            plan.representation_ir.to_json()
        ),
        "target_lowering_digest": contract_digest(plan.target_lowering),
        "relowering_request_digest": contract_digest(
            _relowering_request(plan)
        ),
    }
    if any(
        record_document[field] != digest
        for field, digest in expected_digests.items()
    ):
        raise ModelCompileError(
            "published candidate evidence does not match requested plan"
        )
    record_ref = (f"records/{integrity['construction_id']}.json",)
    next_session = session.transition_candidate(
        plan.candidate_id,
        CandidateState.STAGED,
        evidence_refs=record_ref,
        reason="recovered fully published candidate after interrupted staging",
    )
    return CandidateConstructionOutcome(
        status="completed",
        record=loaded.record,
        session=next_session,
        staged_candidate_path=loaded.path,
    )


def _remove_abandoned_staging(
    workspace_root: Path,
    candidate_id: str,
) -> None:
    staging_root = workspace_root / ".staging"
    if staging_root.is_symlink():
        raise ModelCompileError("candidate staging directory must not be a symlink")
    if staging_root.exists():
        removed = False
        for path in staging_root.glob(f"{candidate_id}.*"):
            if path.is_symlink() or not path.is_dir():
                raise ModelCompileError(
                    f"abandoned candidate staging path is unsafe: {path}"
                )
            _remove_candidate_tree(path)
            removed = True
        if removed:
            _fsync_directory(staging_root)
    record_staging = workspace_root / "records" / ".staging"
    if record_staging.is_symlink():
        raise ModelCompileError(
            "candidate record staging directory must not be a symlink"
        )
    if record_staging.exists():
        for path in record_staging.glob(f"{candidate_id}.*"):
            if path.is_symlink() or not path.is_file():
                raise ModelCompileError(
                    f"abandoned candidate record path is unsafe: {path}"
                )
            path.unlink()
        _fsync_directory(record_staging)


def _construct_candidate(
    *,
    package_dir: Path,
    workspace_root: Path,
    plan: ProviderCandidatePlan,
    session: OptimizationSession,
    candidate: Json,
    build_plan: CandidateBuildPlan,
    source_seal: Json,
    started_ns: int,
    initial_rss: int,
    semantic_constructor: CandidateSemanticConstructor,
    ordinary_relowerer: CandidateOrdinaryRelowerer,
    physical_optimizer: CandidatePhysicalOptimizer,
    artifact_validators: ArtifactValidatorRegistry | None,
    cancel_requested: Callable[[], bool] | None,
) -> CandidateConstructionOutcome:
    candidate_id = plan.candidate_id

    ready_path = workspace_root / "ready" / candidate_id
    staging_identity = f"{candidate_id}.{uuid4().hex}"
    construction_id = stable_contract_id(
        "construction",
        candidate_id,
        contract_digest(plan.representation_ir.to_json()),
        contract_digest(plan.target_lowering),
        staging_identity,
    )
    staging_path = workspace_root / ".staging" / staging_identity
    staging_path.mkdir(parents=True, exist_ok=False)
    _fsync_directory(staging_path.parent)
    context = CandidateConstructionContext(
        package_dir=package_dir,
        staging_dir=staging_path,
        candidate=candidate,
        representation_graph=plan.representation_ir.to_json(),
        target_lowering=plan.target_lowering,
        build_plan=build_plan,
        started_ns=started_ns,
        cancel_requested=cancel_requested,
    )
    relowering_request = _relowering_request(plan)
    phases: list[Json] = []
    diagnostics: list[str] = []
    status = "completed"
    staged_path: Path | None = None
    integrity: Json | None = None
    artifact_records: list[Json] = []
    failure: Exception | None = None
    peak_staging_bytes = context.peak_staging_bytes
    try:
        context.checkpoint()
        for name, document in (
            ("candidate.json", candidate),
            ("representation_graph.json", plan.representation_ir.to_json()),
            ("target_lowering.json", plan.target_lowering),
            ("build_plan.json", build_plan.to_json()),
            ("mount_plan.json", plan.mount_requirements.to_json()),
            ("relowering_request.json", relowering_request),
        ):
            context.write_internal_contract(name, document)
    except ModelCompileCancelled as error:
        status = "cancelled"
        diagnostics.append(str(error))
        failure = error
    except Exception as error:
        status = "failed"
        diagnostics.append(f"{type(error).__name__}: {error}")
        failure = error

    phase_services = (
        (
            CONSTRUCTION_PHASES[0],
            semantic_constructor.construct_semantic_artifacts,
        ),
        (
            CONSTRUCTION_PHASES[1],
            ordinary_relowerer.run_ordinary_lowering,
        ),
        (
            CONSTRUCTION_PHASES[2],
            physical_optimizer.optimize_physical_artifacts,
        ),
    )

    for phase, service in phase_services if failure is None else ():
        phase_started = time.monotonic_ns()
        before_staging = context.staging_bytes
        try:
            context.begin_phase(phase)
            service(context)
            context.end_phase()
            phase_status = "completed"
            phase_diagnostics: list[str] = []
        except ModelCompileCancelled as error:
            phase_status = "cancelled"
            phase_diagnostics = [str(error)]
            status = "cancelled"
            failure = error
        except Exception as error:
            phase_status = "failed"
            phase_diagnostics = [f"{type(error).__name__}: {error}"]
            status = "failed"
            failure = error
        phase_finished = time.monotonic_ns()
        phases.append(
            {
                "name": phase,
                "status": phase_status,
                "started_ns": phase_started - started_ns,
                "finished_ns": phase_finished - started_ns,
                "duration_ns": phase_finished - phase_started,
                "staging_bytes_written": context.staging_bytes - before_staging,
                "peak_temporary_bytes": max(
                    context.peak_transient_bytes,
                    max(0, _resident_bytes() - initial_rss),
                ),
                "diagnostics": phase_diagnostics,
            }
        )
        diagnostics.extend(phase_diagnostics)
        if failure is not None:
            break

    if failure is None:
        try:
            context.validate_complete()
            validators = (
                artifact_validators
                if artifact_validators is not None
                else ArtifactValidatorRegistry.with_builtin_validators()
            )
            validation_results = validators.validate_artifacts(
                staging_path,
                build_plan,
            )
            artifact_records = context.artifact_records(validation_results)
            integrity = write_staged_candidate_integrity(
                staging_path,
                candidate_id=candidate_id,
                construction_id=construction_id,
            )
            context.observe_total_staging_bytes(
                _tree_file_bytes(staging_path)
            )
            peak_staging_bytes = context.peak_staging_bytes
            validate_staged_candidate(
                staging_path,
                expected_candidate_id=candidate_id,
                expected_build_plan=build_plan,
            )
            verify_source_package_seal(package_dir, build_plan, source_seal)
            ready_path.parent.mkdir(parents=True, exist_ok=True)
            staging_path.replace(ready_path)
            _fsync_directory(ready_path.parent)
            validate_staged_candidate(
                ready_path,
                expected_candidate_id=candidate_id,
                expected_build_plan=build_plan,
            )
            verify_source_package_seal(package_dir, build_plan, source_seal)
            staged_path = ready_path
        except ModelCompileCancelled as error:
            status = "cancelled"
            diagnostics.append(str(error))
            failure = error
            phases[-1]["status"] = "cancelled"
            phases[-1]["diagnostics"].append(str(error))
        except Exception as error:
            status = "failed"
            diagnostics.append(f"{type(error).__name__}: {error}")
            failure = error
            phases[-1]["status"] = "failed"
            phases[-1]["diagnostics"].append(
                f"{type(error).__name__}: {error}"
            )

    if failure is not None:
        _remove_candidate_tree(staging_path)
        _remove_candidate_tree(ready_path)
        artifact_records = []
        integrity = None
        try:
            verify_source_package_seal(package_dir, build_plan, source_seal)
        except Exception as seal_error:
            status = "failed"
            diagnostics.append(f"{type(seal_error).__name__}: {seal_error}")
            if phases:
                phases[-1]["status"] = "failed"
                phases[-1]["diagnostics"].append(
                    f"{type(seal_error).__name__}: {seal_error}"
                )

    finished_ns = time.monotonic_ns()
    generated_bytes = sum(record["byte_count"] for record in artifact_records)
    permanent_bytes = sum(record["resident_bytes"] for record in artifact_records)
    record_document = {
        "schema": CANDIDATE_CONSTRUCTION_SCHEMA,
        "construction_id": construction_id,
        "candidate_id": candidate_id,
        "status": status,
        "staging_identity": staging_identity,
        "source_seal": source_seal,
        "representation_graph_digest": contract_digest(
            plan.representation_ir.to_json()
        ),
        "target_lowering_digest": contract_digest(plan.target_lowering),
        "relowering_request_digest": contract_digest(relowering_request),
        "phases": phases,
        "artifacts": artifact_records,
        "integrity": integrity_evidence(integrity) if integrity is not None else None,
        "resource_measurements": {
            "construction_time_ns": finished_ns - started_ns,
            "peak_temporary_bytes": max(
                context.peak_transient_bytes,
                max(0, _resident_bytes() - initial_rss),
            ),
            "peak_staging_bytes": max(
                peak_staging_bytes,
                context.peak_staging_bytes,
            ),
            "final_permanent_bytes": permanent_bytes,
            "generated_artifact_bytes": generated_bytes,
        },
        "diagnostics": diagnostics,
    }
    record = ContractDocument.from_json(
        record_document,
        expected_schema=CANDIDATE_CONSTRUCTION_SCHEMA,
    )
    record_path: Path | None = None
    try:
        record_path = _write_record(workspace_root, record)
        record_ref = (f"records/{construction_id}.json",)
        if status == "completed":
            next_session = session.transition_candidate(
                candidate_id,
                CandidateState.STAGED,
                evidence_refs=record_ref,
                reason="candidate artifacts were constructed and atomically staged",
            )
        elif status == "cancelled":
            next_session = session.transition_candidate(
                candidate_id,
                CandidateState.CANCELLED,
                evidence_refs=(),
                reason="candidate construction was cancelled and cleaned",
            )
        else:
            next_session = session.transition_candidate(
                candidate_id,
                CandidateState.FAILED,
                evidence_refs=record_ref,
                reason="candidate construction failed in isolation and was cleaned",
            )
    except BaseException:
        if staged_path is not None:
            _remove_candidate_tree(staged_path)
        if record_path is not None:
            record_path.unlink(missing_ok=True)
            _fsync_directory(record_path.parent)
        raise
    return CandidateConstructionOutcome(
        status=status,
        record=record,
        session=next_session,
        staged_candidate_path=staged_path,
    )


def _relowering_request(plan: ProviderCandidatePlan) -> Json:
    graph = plan.representation_ir.to_json()
    representations = {
        representation["id"]: representation
        for representation in graph["physical_representations"]
    }
    signals = {signal["id"]: signal for signal in graph["signals"]}
    boundaries = []
    for port in graph["public_ports"]:
        signal = signals[port["signal_id"]]
        representation = representations[signal["physical_representation_id"]]
        boundaries.append(
            {
                "name": port["id"],
                "direction": port["direction"],
                "logical_contract_id": port["logical_contract_id"],
                "physical_representation_id": representation["id"],
                "physical_kind": representation["kind"],
            }
        )
    request = {
        "schema": RELOWERING_REQUEST_SCHEMA,
        "request_id": stable_contract_id(
            "relower",
            plan.candidate_id,
            contract_digest(graph),
        ),
        "candidate_id": plan.candidate_id,
        "scope_ids": graph["scope_ids"],
        "representation_digest": contract_digest(graph),
        "required_passes": [
            "ordinary_lowering",
            "physical_optimization",
        ],
        "boundary_contracts": sorted(
            boundaries,
            key=lambda boundary: boundary["name"],
        ),
    }
    ContractDocument.from_json(
        request,
        expected_schema=RELOWERING_REQUEST_SCHEMA,
    )
    return request


def _validate_candidate_build_binding(plan: ProviderCandidatePlan) -> None:
    candidate = plan.candidate.to_json()
    declared = tuple(
        artifact["path"] for artifact in candidate["artifact_declarations"]
    )
    if plan.construction_requirements.output_paths != declared:
        raise ModelCompileError(
            "candidate build plan outputs do not match artifact declarations"
        )
    graph = plan.representation_ir.to_json()
    graph_artifacts = {
        resource["artifact"]["path"] for resource in graph["resources"]
    }
    graph_artifacts.update(
        kernel["artifact"]["path"] for kernel in graph["physical_kernels"]
    )
    if not graph_artifacts <= set(declared):
        raise ModelCompileError(
            "representation graph references undeclared candidate artifacts"
        )


def _validate_isolation(package_dir: Path, workspace_root: Path) -> None:
    if workspace_root == package_dir or workspace_root.is_relative_to(package_dir):
        raise ModelCompileError(
            "candidate workspace must be outside the immutable source package"
        )
    if package_dir.is_relative_to(workspace_root):
        raise ModelCompileError(
            "candidate workspace must not contain the immutable source package"
        )


def _validate_session(
    session: OptimizationSession,
    candidate: Json,
    package_dir: Path,
) -> None:
    manifest = _read_object(package_dir / "vulkan_resident_package.json")
    if session.package_id != manifest.get("package_id"):
        raise ModelCompileError("optimizer session package does not match source package")
    stage_path = package_dir / str(manifest.get("representation_optimization_path"))
    stage = _read_object(stage_path)
    if session.exact_baseline_digest != stage.get("exact_baseline", {}).get(
        "contract_digest"
    ):
        raise ModelCompileError("optimizer session exact baseline does not match package")
    matching = [
        lifecycle
        for lifecycle in session.candidates
        if lifecycle.candidate_id == candidate["candidate_id"]
    ]
    if len(matching) != 1 or matching[0].state != CandidateState.SYNTHESIZED:
        raise ModelCompileError(
            "candidate must be registered in synthesized state before staging"
        )
    if tuple(candidate["source_contract_digests"]) != matching[0].source_contract_digests:
        raise ModelCompileError(
            "candidate lifecycle source contracts do not match candidate"
        )


def _write_record(workspace_root: Path, record: ContractDocument) -> Path:
    records = workspace_root / "records"
    if records.is_symlink():
        raise ModelCompileError(
            "candidate construction record directory must not be a symlink"
        )
    records.mkdir(parents=True, exist_ok=True)
    path = records / f"{record.to_json()['construction_id']}.json"
    if path.exists() or path.is_symlink():
        raise ModelCompileError("candidate construction record already exists")
    payload = record.to_bytes() + b"\n"
    staging = records / ".staging"
    if staging.is_symlink():
        raise ModelCompileError(
            "candidate record staging directory must not be a symlink"
        )
    staging.mkdir(parents=True, exist_ok=True)
    candidate_id = record.to_json()["candidate_id"]
    temporary = staging / f"{candidate_id}.{uuid4().hex}.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(temporary, flags, 0o644)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(descriptor)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        _fsync_directory(staging)
        raise
    _fsync_directory(records)
    return path


def _read_object(path: Path) -> Json:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ModelCompileError(f"optimizer staging input is unreadable: {path}") from error
    if not isinstance(document, dict):
        raise ModelCompileError(f"optimizer staging input must be an object: {path}")
    return document


def _resident_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage) * 1024


def _tree_file_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ModelCompileError(
                f"candidate staging contains a symbolic link: {path}"
            )
        if path.is_file():
            total += path.stat().st_size
    return total


def _remove_candidate_tree(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
    if path.exists() or path.is_symlink():
        raise ModelCompileError(
            f"candidate workspace cleanup did not complete: {path}"
        )
    if path.parent.exists():
        _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
