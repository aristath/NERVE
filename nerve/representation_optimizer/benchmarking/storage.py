from __future__ import annotations

import json
import os
import shutil
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.benchmarking.contracts import (
    BENCHMARK_EVIDENCE_INTEGRITY_SCHEMA,
    BENCHMARK_RECORD_SCHEMA,
    BenchmarkPlan,
    BenchmarkRun,
    validate_benchmark_record,
)
from nerve.representation_optimizer.benchmarking.protocols import (
    NormalExecutionAdapter,
)
from nerve.representation_optimizer.benchmarking.runner import (
    validate_complete_run_against_plan,
)
from nerve.representation_optimizer.contracts import (
    ContractDocument,
    contract_digest,
)


BENCHMARK_INTEGRITY_FILE = "integrity.json"


def publish_benchmark_evidence(
    workspace_root: Path,
    *,
    plan: BenchmarkPlan,
    run: BenchmarkRun,
    record: ContractDocument,
    trace_source: NormalExecutionAdapter,
) -> Path:
    workspace_root = workspace_root.resolve()
    record_document = record.to_json()
    validate_benchmark_record(record_document)
    _validate_links(plan, run, record_document)
    benchmark_id = record_document["benchmark_id"]
    benchmarks_root = workspace_root / "benchmarks"
    if benchmarks_root.is_symlink():
        raise ModelCompileError(
            "benchmark evidence directory must not be a symlink"
        )
    ready = benchmarks_root / benchmark_id
    if ready.is_symlink():
        raise ModelCompileError("benchmark evidence path must not be a symlink")
    if ready.exists():
        loaded_plan, loaded_run, loaded_record = load_benchmark_evidence(
            workspace_root,
            benchmark_id,
        )
        if (
            loaded_plan != plan
            or loaded_run != run
            or loaded_record != record
        ):
            raise ModelCompileError(
                "benchmark identity is already bound to different evidence"
            )
        return ready
    staging_root = workspace_root / ".benchmark-staging"
    if staging_root.is_symlink():
        raise ModelCompileError(
            "benchmark staging directory must not be a symlink"
        )
    staging = staging_root / f"{benchmark_id}.{uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    _fsync_directory(staging_root)
    _fsync_directory(workspace_root)
    published_ready = False
    try:
        _write_json(staging / "plan.json", plan.to_json())
        _write_json(staging / "raw_run.json", run.to_json())
        _write_json(staging / "record.json", record_document)
        _copy_raw_artifacts(
            staging,
            plan,
            run,
            trace_source,
        )
        integrity = _integrity_document(staging, benchmark_id)
        _write_json(staging / BENCHMARK_INTEGRITY_FILE, integrity)
        _validate_evidence_tree(staging, expected_benchmark_id=benchmark_id)
        _fsync_tree_directories(staging)
        ready.parent.mkdir(parents=True, exist_ok=True)
        _fsync_directory(workspace_root)
        staging.replace(ready)
        published_ready = True
        _fsync_directory(staging_root)
        _fsync_directory(ready.parent)
        _validate_evidence_tree(ready, expected_benchmark_id=benchmark_id)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        if published_ready and ready.exists():
            shutil.rmtree(ready)
        if staging.exists() or ready.exists():
            raise ModelCompileError(
                "failed to clean incomplete benchmark evidence publication"
            )
        raise
    return ready


def load_benchmark_evidence(
    workspace_root: Path,
    benchmark_id: str,
) -> tuple[BenchmarkPlan, BenchmarkRun, ContractDocument]:
    if (
        not benchmark_id.startswith("benchmark_")
        or len(benchmark_id) != len("benchmark_") + 32
        or any(
            character not in "0123456789abcdef"
            for character in benchmark_id.removeprefix("benchmark_")
        )
    ):
        raise ModelCompileError("benchmark evidence identity is invalid")
    benchmarks_root = workspace_root.resolve() / "benchmarks"
    if benchmarks_root.is_symlink():
        raise ModelCompileError(
            "benchmark evidence directory must not be a symlink"
        )
    root = benchmarks_root / benchmark_id
    if root.is_symlink():
        raise ModelCompileError("benchmark evidence path must not be a symlink")
    _validate_evidence_tree(root, expected_benchmark_id=benchmark_id)
    plan = BenchmarkPlan.from_json(_read_object(root / "plan.json"))
    run = BenchmarkRun.from_json(_read_object(root / "raw_run.json"))
    record = ContractDocument.from_json(
        _read_object(root / "record.json"),
        expected_schema=BENCHMARK_RECORD_SCHEMA,
    )
    _validate_links(plan, run, record.to_json())
    return plan, run, record


def _validate_links(
    plan: BenchmarkPlan,
    run: BenchmarkRun,
    record: Json,
) -> None:
    run_document = run.to_json()
    if run_document["status"] == "completed":
        validate_complete_run_against_plan(plan, run)
    traces = {
        trace["path"]
        for observation in run_document["observations"]
        for trace in observation["traces"].values()
    }
    raw = record["raw_evidence"]
    plan_document = plan.to_json()
    observation_by_id = {
        observation["observation_id"]: observation
        for observation in run_document["observations"]
    }
    plan_workload_ids = [
        workload["workload_id"] for workload in plan_document["workloads"]
    ]
    record_workload_ids = [
        workload["workload_id"] for workload in record["workloads"]
    ]
    reproducibility_links_are_valid = all(
        all(
            observation_id in observation_by_id
            and observation_by_id[observation_id]["workload_id"]
            == group["workload_id"]
            and observation_by_id[observation_id]["role"] == group["role"]
            and observation_by_id[observation_id]["seed"] == group["seed"]
            and observation_by_id[observation_id]["order_index"]
            == group["order_index"]
            and observation_by_id[observation_id]["phase"] == "measured"
            for observation_id in group["observation_ids"]
        )
        for group in record["reproducibility"]
    )
    if (
        run_document["plan_id"] != plan.plan_id
        or record["candidate_id"] != plan.candidate_id
        or record["plan_digest"] != contract_digest(plan_document)
        or record["run_digest"] != contract_digest(run_document)
        or record["construction_record_digest"]
        != plan_document["construction_record_digest"]
        or record["reference_implementation_id"]
        != plan_document["implementations"]["reference"]["implementation_id"]
        or record["matched_conditions_digest"]
        != plan_document["matched_conditions_digest"]
        or record_workload_ids != plan_workload_ids
        or not reproducibility_links_are_valid
        or raw["run_id"] != run_document["run_id"]
        or raw["observation_count"] != len(run_document["observations"])
        or raw["residency_event_count"]
        != len(run_document["residency_events"])
        or raw["host_elapsed_sample_count"]
        != len(run_document["host_elapsed_ns"])
        or raw["trace_artifact_count"] != len(traces)
    ):
        raise ModelCompileError(
            "benchmark plan, raw run, and summary record do not match"
        )


def _integrity_document(root: Path, benchmark_id: str) -> Json:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            if path.is_symlink():
                raise ModelCompileError(
                    "benchmark evidence contains a symbolic link"
                )
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "byte_count": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return {
        "schema": BENCHMARK_EVIDENCE_INTEGRITY_SCHEMA,
        "benchmark_id": benchmark_id,
        "files": files,
    }


def _validate_evidence_tree(
    root: Path,
    *,
    expected_benchmark_id: str,
) -> Json:
    integrity = _read_object(root / BENCHMARK_INTEGRITY_FILE)
    if set(integrity) != {"schema", "benchmark_id", "files"}:
        raise ModelCompileError("benchmark integrity fields are invalid")
    if (
        integrity["schema"] != BENCHMARK_EVIDENCE_INTEGRITY_SCHEMA
        or integrity["benchmark_id"] != expected_benchmark_id
    ):
        raise ModelCompileError("benchmark integrity identity is invalid")
    records = integrity["files"]
    if not isinstance(records, list):
        raise ModelCompileError("benchmark integrity files must be a list")
    paths = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "byte_count",
            "sha256",
        }:
            raise ModelCompileError("benchmark integrity record is malformed")
        if (
            isinstance(record["byte_count"], bool)
            or not isinstance(record["byte_count"], int)
            or record["byte_count"] < 0
            or not isinstance(record["sha256"], str)
            or len(record["sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in record["sha256"]
            )
        ):
            raise ModelCompileError(
                "benchmark integrity size or digest is malformed"
            )
        relative = Path(str(record["path"]))
        if (
            relative.is_absolute()
            or "." in relative.parts
            or ".." in relative.parts
            or relative.as_posix() != record["path"]
        ):
            raise ModelCompileError("benchmark integrity path is unsafe")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ModelCompileError(
                "benchmark integrity references a non-regular file"
            )
        if (
            record["byte_count"] != path.stat().st_size
            or record["sha256"] != _file_sha256(path)
        ):
            raise ModelCompileError(
                f"benchmark evidence failed integrity: {record['path']!r}"
            )
        paths.append(record["path"])
    if paths != sorted(set(paths)):
        raise ModelCompileError(
            "benchmark integrity paths must be sorted and unique"
        )
    actual = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / BENCHMARK_INTEGRITY_FILE
    )
    if actual != paths or any(path.is_symlink() for path in root.rglob("*")):
        raise ModelCompileError(
            "benchmark evidence tree contains unrecorded entries"
        )
    plan = _read_object(root / "plan.json")
    run = _read_object(root / "raw_run.json")
    fixture_paths = _fixture_paths_from_plan_document(plan)
    expected_paths = sorted(
        {
            "plan.json",
            "raw_run.json",
            "record.json",
            *fixture_paths,
            *(
                trace["path"]
                for observation in run.get("observations", [])
                if isinstance(observation, dict)
                for trace in (
                    observation.get("traces", {}).values()
                    if isinstance(observation.get("traces"), dict)
                    else ()
                )
                if isinstance(trace, dict) and isinstance(trace.get("path"), str)
            ),
        }
    )
    if paths != expected_paths:
        raise ModelCompileError(
            "benchmark integrity does not exactly cover evidence files"
        )
    return integrity


def _copy_raw_artifacts(
    root: Path,
    plan: BenchmarkPlan,
    run: BenchmarkRun,
    source: NormalExecutionAdapter,
) -> None:
    trace_refs = {
        trace["path"]: trace["digest"]
        for observation in run.to_json()["observations"]
        for trace in observation["traces"].values()
    }
    fixture_refs: dict[str, str] = {}
    for workload in plan.to_json()["workloads"]:
        basis = workload["useful_work"]["output_allowance_basis"]
        references = [
            workload["input"],
            workload["initial_state"],
            basis.get("artifact"),
        ]
        for reference in references:
            if reference is None:
                continue
            previous = fixture_refs.setdefault(
                reference["path"],
                reference["digest"],
            )
            if previous != reference["digest"]:
                raise ModelCompileError(
                    "benchmark fixture path is bound to different digests"
                )
    overlap = set(trace_refs) & set(fixture_refs)
    if overlap:
        raise ModelCompileError(
            "benchmark trace and fixture artifact namespaces overlap"
        )
    artifacts = {
        **fixture_refs,
        **trace_refs,
    }
    for relative_path, expected_digest in sorted(artifacts.items()):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        digest = sha256()
        descriptor = os.open(path, flags, 0o644)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                chunks = (
                    source.iter_trace_artifact(relative_path)
                    if relative_path in trace_refs
                    else source.iter_fixture_artifact(
                        relative_path,
                        candidate_id=plan.candidate_id,
                    )
                )
                for chunk in chunks:
                    if not isinstance(chunk, bytes):
                        raise ModelCompileError(
                            "benchmark raw artifact source yielded a "
                            "non-byte chunk"
                        )
                    stream.write(chunk)
                    digest.update(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(descriptor)
        actual = (
            "nerve.optimizer.artifact_sha256.v1:"
            f"{digest.hexdigest()}"
        )
        if actual != expected_digest:
            raise ModelCompileError(
                "benchmark raw artifact changed before publication: "
                f"{relative_path!r}"
            )
        _fsync_directory(path.parent)


def _fixture_paths_from_plan_document(document: Json) -> set[str]:
    paths = set()
    workloads = document.get("workloads")
    if not isinstance(workloads, list):
        return paths
    for workload in workloads:
        if not isinstance(workload, dict):
            continue
        references = [workload.get("input"), workload.get("initial_state")]
        useful = workload.get("useful_work")
        if isinstance(useful, dict):
            basis = useful.get("output_allowance_basis")
            if isinstance(basis, dict):
                references.append(basis.get("artifact"))
        for reference in references:
            if (
                isinstance(reference, dict)
                and isinstance(reference.get("path"), str)
            ):
                paths.add(reference["path"])
    return paths


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, document: Json) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            encoder = json.JSONEncoder(
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            for chunk in encoder.iterencode(document):
                stream.write(chunk.encode("utf-8"))
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _read_object(path: Path) -> Json:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ModelCompileError(
            f"benchmark evidence is unreadable: {path}"
        ) from error
    if not isinstance(document, dict):
        raise ModelCompileError(
            f"benchmark evidence must be an object: {path}"
        )
    return document


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree_directories(root: Path) -> None:
    directories = [
        path
        for path in root.rglob("*")
        if path.is_dir() and not path.is_symlink()
    ]
    for path in sorted(
        directories,
        key=lambda directory: len(directory.relative_to(root).parts),
        reverse=True,
    ):
        _fsync_directory(path)
    _fsync_directory(root)
