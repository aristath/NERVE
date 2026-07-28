from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import (
    CANDIDATE_CONSTRUCTION_SCHEMA,
    ContractDocument,
    contract_digest,
)
from nerve.representation_optimizer.staging.contracts import (
    CandidateBuildPlan,
    SOURCE_PACKAGE_SEAL_FILE,
    staged_file_digest,
)
from nerve.representation_optimizer.staging.integrity import (
    integrity_evidence,
    validate_staged_candidate,
)
from nerve.representation_optimizer.staging.source_seal import (
    verify_source_package_seal,
)


@dataclass(frozen=True)
class LoadedStagedCandidate:
    path: Path
    record: ContractDocument
    build_plan: CandidateBuildPlan


def load_staged_candidate(
    workspace_root: Path,
    candidate_id: str,
    *,
    package_dir: Path | None = None,
) -> LoadedStagedCandidate:
    if (
        not candidate_id.startswith("candidate_")
        or len(candidate_id) != len("candidate_") + 32
    ):
        raise ModelCompileError("staged candidate_id is invalid")
    workspace_root = workspace_root.resolve()
    published_path = workspace_root / "ready" / candidate_id
    if published_path.is_symlink():
        raise ModelCompileError("published candidate must not be a symlink")
    root = published_path.resolve()
    integrity = validate_staged_candidate(
        root,
        expected_candidate_id=candidate_id,
    )
    construction_id = integrity["construction_id"]
    record_path = workspace_root / "records" / f"{construction_id}.json"
    if record_path.is_symlink():
        raise ModelCompileError(
            "candidate construction record must not be a symlink"
        )
    try:
        record_payload = record_path.read_bytes()
    except OSError as error:
        raise ModelCompileError(
            "candidate construction record is unreadable"
        ) from error
    record = ContractDocument.from_bytes(
        record_payload,
        expected_schema=CANDIDATE_CONSTRUCTION_SCHEMA,
    )
    document = record.to_json()
    if (
        document["candidate_id"] != candidate_id
        or document["construction_id"] != construction_id
        or document["status"] != "completed"
    ):
        raise ModelCompileError(
            "candidate construction record does not match staged candidate"
        )
    if document["integrity"] != integrity_evidence(integrity):
        raise ModelCompileError(
            "candidate construction record integrity evidence does not match"
        )
    build_plan = CandidateBuildPlan.from_json(
        _read_object(root / "contracts" / "build_plan.json")
    )
    validate_staged_candidate(
        root,
        expected_candidate_id=candidate_id,
        expected_build_plan=build_plan,
    )
    artifact_by_path = {
        artifact["path"]: artifact for artifact in document["artifacts"]
    }
    if set(artifact_by_path) != set(build_plan.output_paths):
        raise ModelCompileError(
            "candidate construction record does not cover every planned output"
        )
    for relative_path, artifact in artifact_by_path.items():
        path = root / relative_path
        if (
            artifact["byte_count"] != path.stat().st_size
            or artifact["digest"] != staged_file_digest(path)
            or artifact["validation"]["status"] != "passed"
        ):
            raise ModelCompileError(
                f"candidate construction evidence is invalid: {relative_path!r}"
            )
    contract_digests = {
        "representation_graph_digest": contract_digest(
            _read_object(root / "contracts" / "representation_graph.json")
        ),
        "target_lowering_digest": contract_digest(
            _read_object(root / "contracts" / "target_lowering.json")
        ),
        "relowering_request_digest": contract_digest(
            _read_object(root / "contracts" / "relowering_request.json")
        ),
    }
    if any(document[field] != digest for field, digest in contract_digests.items()):
        raise ModelCompileError(
            "candidate construction record contract digest does not match"
        )
    if (
        _read_object(root / "contracts" / SOURCE_PACKAGE_SEAL_FILE)
        != document["source_seal"]
    ):
        raise ModelCompileError(
            "candidate source package seal does not match construction evidence"
        )
    if package_dir is not None:
        verify_source_package_seal(
            package_dir,
            build_plan,
            document["source_seal"],
        )
    return LoadedStagedCandidate(
        path=root,
        record=record,
        build_plan=build_plan,
    )


def _read_object(path: Path) -> Json:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ModelCompileError(f"staged candidate contract is unreadable: {path}") from error
    if not isinstance(document, dict):
        raise ModelCompileError(f"staged candidate contract must be an object: {path}")
    return document
