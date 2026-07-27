from __future__ import annotations

import errno
import fcntl
import json
import os
import shutil
from pathlib import Path
from typing import Callable
from uuid import uuid4

from nerve.compilation import Json, ModelCompileError, check_compile_cancelled
from nerve.model_package_integrity import (
    build_package_artifact_integrity,
)
from nerve.model_package_validation import validate_compiled_package
from nerve.representation_optimizer.analysis.evidence import (
    validate_analysis_run_directory,
)
from nerve.representation_optimizer.benchmarking.storage import (
    load_benchmark_evidence,
)
from nerve.representation_optimizer.contracts import (
    canonical_json_bytes,
    contract_digest,
)
from nerve.representation_optimizer.lifecycle import (
    CandidateLifecycle,
    CandidateState,
    OptimizationSession,
)
from nerve.representation_optimizer.promotion.contracts import (
    ImplementationRegistry,
    append_implementation_registry_entries,
)
from nerve.representation_optimizer.promotion.orchestrator import (
    PreparedPromotion,
)
from nerve.representation_optimizer.stage import (
    OPTIMIZER_IMPLEMENTATION_REGISTRY_FILE,
    OPTIMIZER_STAGE_FILE,
    load_optimizer_stage,
    validate_optimizer_stage,
)
from nerve.representation_optimizer.staging.integrity import (
    integrity_evidence,
    validate_staged_candidate,
)
from nerve.representation_optimizer.validation.storage import (
    load_prebenchmark_evidence,
    load_validation_evidence,
)


_FICLONE = 0x40049409
_PACKAGE_MANIFEST_FILE = "vulkan_resident_package.json"


def publish_promoted_package(
    *,
    source_package_dir: Path,
    destination_package_dir: Path,
    promotions: tuple[PreparedPromotion, ...],
    session: OptimizationSession,
    cancel_requested: Callable[[], bool] | None = None,
) -> Path:
    check_compile_cancelled(cancel_requested)
    if not promotions:
        raise ModelCompileError(
            "optimized package publication requires at least one promotion"
        )
    source = source_package_dir.resolve()
    destination = destination_package_dir.resolve()
    if source == destination:
        raise ModelCompileError(
            "optimized package publication requires a distinct destination"
        )
    if destination.exists() or destination.is_symlink():
        raise ModelCompileError(
            "optimized package destination already exists"
        )
    if not source.is_dir() or source.is_symlink():
        raise ModelCompileError(
            "source compiled package must be a regular directory"
        )
    manifest = _read_object(source / _PACKAGE_MANIFEST_FILE)
    validate_compiled_package(source, manifest)
    check_compile_cancelled(cancel_requested)
    source_stage_path = (
        source / "optimization" / OPTIMIZER_STAGE_FILE
    )
    source_stage = load_optimizer_stage(
        source_stage_path,
        package_dir=source,
    )
    _validate_publication_session(source_stage, session, promotions)
    _revalidate_promotions(
        source,
        promotions,
        cancel_requested=cancel_requested,
    )
    check_compile_cancelled(cancel_requested)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = destination.parent / ".nerve-package-staging"
    if staging_root.is_symlink():
        raise ModelCompileError(
            "package publication staging root must not be a symlink"
        )
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = staging_root / f"{destination.name}.{uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    _fsync_directory(staging_root)
    published = False
    try:
        _clone_tree_contents(
            source,
            staging,
            cancel_requested=cancel_requested,
        )
        registry_path = (
            staging
            / "optimization"
            / OPTIMIZER_IMPLEMENTATION_REGISTRY_FILE
        )
        registry = ImplementationRegistry.from_json(
            _read_object(registry_path)
        )
        for promotion in promotions:
            check_compile_cancelled(cancel_requested)
            _publish_implementation_bundle(
                staging,
                promotion,
                cancel_requested=cancel_requested,
            )
        registry = append_implementation_registry_entries(
            registry,
            (
                promotion.registry_entry
                for promotion in promotions
            ),
        )
        _write_json(registry_path, registry.to_json())

        published_session = _build_published_session(
            source_stage,
            session,
            promotions,
        )
        stage_path = (
            staging / "optimization" / OPTIMIZER_STAGE_FILE
        )
        stage = _read_object(stage_path)
        stage["status"] = "optimized"
        stage["session"] = published_session.to_json()
        stage["implementation_registry"] = {
            "artifact_ref": (
                "optimization/"
                f"{OPTIMIZER_IMPLEMENTATION_REGISTRY_FILE}"
            ),
            "contract_digest": contract_digest(registry.to_json()),
            "implementation_count": len(registry.implementations),
        }
        _write_json(stage_path, stage)
        validate_optimizer_stage(stage, package_dir=staging)

        staged_manifest = _read_object(
            staging / _PACKAGE_MANIFEST_FILE
        )
        staged_manifest["artifact_integrity"] = (
            build_package_artifact_integrity(staging)
        )
        _write_json(
            staging / _PACKAGE_MANIFEST_FILE,
            staged_manifest,
        )
        validate_compiled_package(staging, staged_manifest)
        check_compile_cancelled(cancel_requested)
        _fsync_tree(
            staging,
            cancel_requested=cancel_requested,
        )
        # This is the publication commit point. A cancellation observed before
        # it leaves no destination; a signal arriving after it applies to the
        # now-complete package rather than rewriting an atomic publication.
        check_compile_cancelled(cancel_requested)
        os.replace(staging, destination)
        published = True
        _fsync_directory(staging_root)
        _fsync_directory(destination.parent)

        published_manifest = _read_object(
            destination / _PACKAGE_MANIFEST_FILE
        )
        validate_compiled_package(destination, published_manifest)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        if published and destination.exists():
            shutil.rmtree(destination)
        if staging.exists() or destination.exists():
            raise ModelCompileError(
                "failed to clean incomplete optimized package publication"
            )
        raise
    finally:
        _remove_empty_staging_root(staging_root)
    return destination


def _publish_implementation_bundle(
    package: Path,
    promotion: PreparedPromotion,
    *,
    cancel_requested: Callable[[], bool] | None,
) -> None:
    check_compile_cancelled(cancel_requested)
    entry = promotion.registry_entry
    root = _package_path(
        package,
        entry["artifact_bundle"]["root_ref"],
        "implementation artifact root",
    )
    if root.exists() or root.is_symlink():
        raise ModelCompileError(
            "implementation artifact root already exists"
        )
    root.mkdir(parents=True, exist_ok=False)
    _clone_tree(
        promotion.staged_candidate.path,
        root / "candidate",
        cancel_requested=cancel_requested,
    )
    _write_json(
        root / "construction_record.json",
        promotion.construction_record.to_json(),
    )
    benchmark_id = promotion.benchmark_record.to_json()[
        "benchmark_id"
    ]
    validation_id = promotion.validation_record.to_json()[
        "validation_id"
    ]
    evidence_root = root / "evidence"
    prebenchmark_id = promotion.prebenchmark_record.to_json()[
        "prebenchmark_id"
    ]
    _clone_tree(
        promotion.prebenchmark_evidence_path,
        evidence_root / "prebenchmark" / prebenchmark_id,
        cancel_requested=cancel_requested,
    )
    _clone_tree(
        promotion.benchmark_evidence_path,
        evidence_root / "benchmarks" / benchmark_id,
        cancel_requested=cancel_requested,
    )
    _clone_tree(
        promotion.validation_evidence_path,
        evidence_root / "validations" / validation_id,
        cancel_requested=cancel_requested,
    )
    for prepared_run in promotion.analysis_runs:
        check_compile_cancelled(cancel_requested)
        _clone_tree(
            prepared_run.path,
            (
                evidence_root
                / "analysis"
                / prepared_run.run.run_id
            ),
            cancel_requested=cancel_requested,
        )
    for profile in promotion.hardware_profiles:
        check_compile_cancelled(cancel_requested)
        _write_json(
            (
                evidence_root
                / "hardware"
                / f"{profile['profile_id']}.json"
            ),
            profile,
        )
    _write_json(
        root / "promotion.json",
        promotion.decision.to_json(),
    )


def _revalidate_promotions(
    source_package: Path,
    promotions: tuple[PreparedPromotion, ...],
    *,
    cancel_requested: Callable[[], bool] | None,
) -> None:
    implementation_ids = [
        promotion.implementation_id for promotion in promotions
    ]
    candidate_ids = [
        promotion.candidate_plan.candidate_id
        for promotion in promotions
    ]
    if (
        implementation_ids != sorted(set(implementation_ids))
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        raise ModelCompileError(
            "promotions must be sorted by implementation and unique by candidate"
        )
    for promotion in promotions:
        check_compile_cancelled(cancel_requested)
        integrity = validate_staged_candidate(
            promotion.staged_candidate.path,
            expected_candidate_id=(
                promotion.candidate_plan.candidate_id
            ),
            expected_build_plan=(
                promotion.candidate_plan.construction_requirements
            ),
        )
        if (
            integrity_evidence(integrity)
            != promotion.decision.to_json()["artifact_integrity"]
        ):
            raise ModelCompileError(
                "promotion candidate artifacts changed before publication"
            )
        benchmark_workspace = (
            promotion.benchmark_evidence_path.parent.parent
        )
        _, _, benchmark = load_benchmark_evidence(
            benchmark_workspace,
            promotion.benchmark_record.to_json()["benchmark_id"],
        )
        validation_workspace = (
            promotion.validation_evidence_path.parent.parent
        )
        *_, validation = load_validation_evidence(
            validation_workspace,
            promotion.validation_record.to_json()["validation_id"],
        )
        if (
            benchmark != promotion.benchmark_record
            or validation != promotion.validation_record
        ):
            raise ModelCompileError(
                "promotion evidence changed before package publication"
            )
        prebenchmark_workspace = (
            promotion.prebenchmark_evidence_path.parent.parent
        )
        (
            _prebenchmark_plan,
            prebenchmark,
            _sanity_run,
        ) = load_prebenchmark_evidence(
            prebenchmark_workspace,
            promotion.prebenchmark_record.to_json()[
                "prebenchmark_id"
            ],
        )
        if prebenchmark != promotion.prebenchmark_record:
            raise ModelCompileError(
                "promotion prebenchmark evidence changed before publication"
            )
        for prepared_run in promotion.analysis_runs:
            check_compile_cancelled(cancel_requested)
            loaded_run = validate_analysis_run_directory(
                prepared_run.path
            )
            if (
                loaded_run != prepared_run.run
                or contract_digest(loaded_run.document)
                != prepared_run.run_digest
            ):
                raise ModelCompileError(
                    "promotion analysis evidence changed before publication"
                )
        # Re-check the source seal after every long validation/benchmark phase.
        from nerve.representation_optimizer.staging.loading import (
            load_staged_candidate,
        )

        loaded = load_staged_candidate(
            promotion.staged_candidate.path.parent.parent,
            promotion.candidate_plan.candidate_id,
            package_dir=source_package,
        )
        if loaded.record != promotion.construction_record:
            raise ModelCompileError(
                "promotion construction record changed before publication"
            )
    check_compile_cancelled(cancel_requested)


def _validate_publication_session(
    source_stage: Json,
    session: OptimizationSession,
    promotions: tuple[PreparedPromotion, ...],
) -> None:
    source_session = OptimizationSession.from_json(
        source_stage["session"]
    )
    if (
        session.package_id != source_session.package_id
        or session.exact_baseline_digest
        != source_session.exact_baseline_digest
    ):
        raise ModelCompileError(
            "promotion session does not belong to the source package"
        )
    source_candidates = {
        candidate.candidate_id: candidate
        for candidate in source_session.candidates
    }
    supplied_candidates = {
        candidate.candidate_id: candidate
        for candidate in session.candidates
    }
    if any(
        supplied_candidates.get(candidate_id) != candidate
        for candidate_id, candidate in source_candidates.items()
    ):
        raise ModelCompileError(
            "promotion session rewrites existing package candidate history"
        )
    promotion_ids = {
        promotion.candidate_plan.candidate_id
        for promotion in promotions
    }
    if promotion_ids.intersection(source_candidates):
        raise ModelCompileError(
            "an implementation candidate cannot be published twice"
        )
    if any(
        supplied_candidates.get(candidate_id) is None
        or supplied_candidates[candidate_id].state
        != CandidateState.PROMOTABLE
        for candidate_id in promotion_ids
    ):
        raise ModelCompileError(
            "every package promotion must be in promotable state"
        )


def _build_published_session(
    source_stage: Json,
    supplied_session: OptimizationSession,
    promotions: tuple[PreparedPromotion, ...],
) -> OptimizationSession:
    source_session = OptimizationSession.from_json(
        source_stage["session"]
    )
    supplied = {
        candidate.candidate_id: candidate
        for candidate in supplied_session.candidates
    }
    document = source_session.to_json()
    for promotion in promotions:
        candidate_id = promotion.candidate_plan.candidate_id
        lifecycle = supplied[candidate_id]
        entry = promotion.registry_entry
        evidence = entry["evidence"]
        state_evidence = {
            CandidateState.STAGED.value: entry["artifact_bundle"][
                "candidate_integrity_ref"
            ],
            CandidateState.STATICALLY_VALIDATED.value: entry[
                "artifact_bundle"
            ]["candidate_integrity_ref"],
            CandidateState.PREBENCHMARK_VALIDATED.value: evidence[
                "prebenchmark_record_ref"
            ],
            CandidateState.BENCHMARKED.value: evidence[
                "benchmark_record_ref"
            ],
            CandidateState.BEHAVIORALLY_VALIDATED.value: evidence[
                "validation_record_ref"
            ],
            CandidateState.PROMOTABLE.value: evidence[
                "promotion_decision_ref"
            ],
        }
        rewritten = lifecycle.to_json()
        rewritten["history"] = [
            {
                **event,
                "evidence_refs": [state_evidence[event["to"]]],
            }
            for event in rewritten["history"]
        ]
        CandidateLifecycle.from_json(rewritten)
        document["candidates"].append(rewritten)
    published = OptimizationSession.from_json(document)
    for promotion in promotions:
        published = published.transition_candidate(
            promotion.candidate_plan.candidate_id,
            CandidateState.PUBLISHED,
            evidence_refs=(
                promotion.registry_entry["evidence"][
                    "promotion_decision_ref"
                ],
            ),
            reason=promotion.decision.to_json()["reason"],
        )
    return published


def _clone_tree(
    source: Path,
    destination: Path,
    *,
    cancel_requested: Callable[[], bool] | None,
) -> None:
    check_compile_cancelled(cancel_requested)
    if destination.exists() or destination.is_symlink():
        raise ModelCompileError(
            f"publication destination already exists: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=False)
    _clone_tree_contents(
        source,
        destination,
        cancel_requested=cancel_requested,
    )


def _clone_tree_contents(
    source: Path,
    destination: Path,
    *,
    cancel_requested: Callable[[], bool] | None,
) -> None:
    check_compile_cancelled(cancel_requested)
    if source.is_symlink() or not source.is_dir():
        raise ModelCompileError(
            f"publication source is not a regular directory: {source}"
    )
    for child in sorted(source.iterdir(), key=lambda path: path.name):
        check_compile_cancelled(cancel_requested)
        if child.is_symlink():
            raise ModelCompileError(
                f"package publication refuses symbolic link: {child}"
            )
        target = destination / child.name
        if child.is_dir():
            target.mkdir(exist_ok=False)
            _clone_tree_contents(
                child,
                target,
                cancel_requested=cancel_requested,
            )
        elif child.is_file():
            _clone_regular_file(
                child,
                target,
                cancel_requested=cancel_requested,
            )
        else:
            raise ModelCompileError(
                f"package publication refuses special file: {child}"
            )


def _clone_regular_file(
    source: Path,
    destination: Path,
    *,
    cancel_requested: Callable[[], bool] | None,
) -> None:
    check_compile_cancelled(cancel_requested)
    source_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        destination_flags |= os.O_NOFOLLOW
    source_fd = os.open(source, source_flags)
    destination_fd = os.open(
        destination,
        destination_flags,
        os.fstat(source_fd).st_mode & 0o777,
    )
    try:
        cloned = False
        try:
            fcntl.ioctl(destination_fd, _FICLONE, source_fd)
            cloned = True
        except OSError as error:
            if error.errno not in {
                errno.EINVAL,
                errno.ENOSYS,
                errno.ENOTTY,
                errno.EOPNOTSUPP,
                errno.EXDEV,
            }:
                raise
        if not cloned:
            os.ftruncate(destination_fd, 0)
            os.lseek(destination_fd, 0, os.SEEK_SET)
            with (
                os.fdopen(os.dup(source_fd), "rb") as reader,
                os.fdopen(os.dup(destination_fd), "wb") as writer,
            ):
                while chunk := reader.read(8 * 1024 * 1024):
                    check_compile_cancelled(cancel_requested)
                    writer.write(chunk)
                writer.flush()
        check_compile_cancelled(cancel_requested)
        os.fsync(destination_fd)
    finally:
        os.close(destination_fd)
        os.close(source_fd)


def _write_json(path: Path, document: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(document) + b"\n"
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _read_object(path: Path) -> Json:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ModelCompileError(
            f"package publication artifact is unreadable: {path}"
        ) from error
    if not isinstance(document, dict):
        raise ModelCompileError(
            f"package publication artifact must be an object: {path}"
        )
    return document


def _package_path(package: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ModelCompileError(f"{label} path must not be empty")
    relative = Path(value)
    if (
        relative.is_absolute()
        or "." in relative.parts
        or ".." in relative.parts
        or relative.as_posix() != value
    ):
        raise ModelCompileError(
            f"{label} must be a canonical package-relative path"
        )
    path = package / relative
    try:
        path.resolve().relative_to(package.resolve())
    except ValueError as error:
        raise ModelCompileError(
            f"{label} escapes the compiled package"
        ) from error
    return path


def _fsync_tree(
    root: Path,
    *,
    cancel_requested: Callable[[], bool] | None,
) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        check_compile_cancelled(cancel_requested)
        if path.is_symlink():
            raise ModelCompileError(
                f"published package contains a symbolic link: {path}"
            )
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        elif path.is_dir():
            _fsync_directory(path)
    check_compile_cancelled(cancel_requested)
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_empty_staging_root(path: Path) -> None:
    try:
        path.rmdir()
    except FileNotFoundError:
        return
    except OSError as error:
        if error.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
            raise
