from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nerve.compilation import Json, ModelCompileError, read_json, write_json
from nerve.representation_optimizer.benchmarking.contracts import (
    BENCHMARK_EVIDENCE_INTEGRITY_SCHEMA,
    BENCHMARK_OBSERVATION_SCHEMA,
    BENCHMARK_PLAN_SCHEMA,
    BENCHMARK_RESIDENCY_EVENT_SCHEMA,
    BENCHMARK_RUN_SCHEMA,
    BENCHMARK_WORKLOAD_SCHEMA,
)
from nerve.representation_optimizer.contracts import (
    ALGEBRAIC_EVIDENCE_SCHEMA,
    BENCHMARK_RECORD_SCHEMA,
    CANDIDATE_CONSTRUCTION_SCHEMA,
    HARDWARE_PROCESS_PROFILE_SCHEMA,
    OPTIMIZATION_SCOPE_CATALOG_SCHEMA,
    OPTIMIZATION_SCOPE_SCHEMA,
    PROMOTION_DECISION_SCHEMA,
    RELOWERING_REQUEST_SCHEMA,
    REPRESENTATION_CANDIDATE_SCHEMA,
    REPRESENTATION_DESCRIPTOR_SCHEMA,
    SOURCE_BEHAVIOR_CONTRACT_SCHEMA,
    VALIDATION_RECORD_SCHEMA,
    contract_digest,
)
from nerve.representation_optimizer.lifecycle import OptimizationSession
from nerve.representation_optimizer.promotion.contracts import (
    IMPLEMENTATION_REGISTRY_SCHEMA,
    RUNTIME_IMPLEMENTATION_PREDICATE_SCHEMA,
    ImplementationRegistry,
    create_empty_implementation_registry,
)
from nerve.representation_optimizer.promotion.package_validation import (
    validate_published_implementation_registry,
)
from nerve.representation_optimizer.representation_ir import (
    REPRESENTATION_GRAPH_SCHEMA,
)
from nerve.representation_optimizer.scope_enumeration.catalog import (
    load_optimization_scope_catalog,
    write_optimization_scope_catalog,
)
from nerve.representation_optimizer.staging.contracts import (
    CANDIDATE_BUILD_PLAN_SCHEMA,
    SOURCE_PACKAGE_SEAL_SCHEMA,
)
from nerve.representation_optimizer.staging.integrity import (
    STAGED_CANDIDATE_INTEGRITY_SCHEMA,
)
from nerve.representation_optimizer.validation.contracts import (
    BEHAVIORAL_ERROR_CONTRACT_SCHEMA,
    PREBENCHMARK_RECORD_SCHEMA,
    PROOF_RESULT_SCHEMA,
    VALIDATION_EVIDENCE_INTEGRITY_SCHEMA,
    VALIDATION_OBSERVATION_SCHEMA,
    VALIDATION_PLAN_SCHEMA,
    VALIDATION_REQUIREMENTS_SCHEMA,
    VALIDATION_ROLE_RESULT_SCHEMA,
    VALIDATION_RESIDENCY_EVENT_SCHEMA,
    VALIDATION_RUN_SCHEMA,
)


OPTIMIZER_STAGE_SCHEMA = "nerve.optimizer.stage.v3"
OPTIMIZER_STAGE_DIR = "optimization"
OPTIMIZER_STAGE_FILE = "stage.json"
OPTIMIZER_IMPLEMENTATION_REGISTRY_FILE = "implementations.json"
OPTIMIZER_CONTRACT_SCHEMAS = (
    OPTIMIZATION_SCOPE_SCHEMA,
    OPTIMIZATION_SCOPE_CATALOG_SCHEMA,
    SOURCE_BEHAVIOR_CONTRACT_SCHEMA,
    ALGEBRAIC_EVIDENCE_SCHEMA,
    HARDWARE_PROCESS_PROFILE_SCHEMA,
    REPRESENTATION_DESCRIPTOR_SCHEMA,
    REPRESENTATION_CANDIDATE_SCHEMA,
    REPRESENTATION_GRAPH_SCHEMA,
    CANDIDATE_BUILD_PLAN_SCHEMA,
    CANDIDATE_CONSTRUCTION_SCHEMA,
    SOURCE_PACKAGE_SEAL_SCHEMA,
    STAGED_CANDIDATE_INTEGRITY_SCHEMA,
    BENCHMARK_WORKLOAD_SCHEMA,
    BENCHMARK_PLAN_SCHEMA,
    BENCHMARK_OBSERVATION_SCHEMA,
    BENCHMARK_RESIDENCY_EVENT_SCHEMA,
    BENCHMARK_RUN_SCHEMA,
    BENCHMARK_RECORD_SCHEMA,
    BENCHMARK_EVIDENCE_INTEGRITY_SCHEMA,
    BEHAVIORAL_ERROR_CONTRACT_SCHEMA,
    VALIDATION_REQUIREMENTS_SCHEMA,
    VALIDATION_PLAN_SCHEMA,
    PROOF_RESULT_SCHEMA,
    VALIDATION_ROLE_RESULT_SCHEMA,
    VALIDATION_OBSERVATION_SCHEMA,
    VALIDATION_RESIDENCY_EVENT_SCHEMA,
    VALIDATION_RUN_SCHEMA,
    PREBENCHMARK_RECORD_SCHEMA,
    VALIDATION_EVIDENCE_INTEGRITY_SCHEMA,
    VALIDATION_RECORD_SCHEMA,
    RUNTIME_IMPLEMENTATION_PREDICATE_SCHEMA,
    PROMOTION_DECISION_SCHEMA,
    IMPLEMENTATION_REGISTRY_SCHEMA,
    RELOWERING_REQUEST_SCHEMA,
)


@dataclass(frozen=True)
class OptimizerStageArtifact:
    path: Path
    document: Json

    def package_reference(self, package_dir: Path) -> str:
        return self.path.relative_to(package_dir).as_posix()


def initialize_optimizer_stage(
    *,
    package_id: str,
    package_dir: Path,
    lowered_index: Json,
    lowered_index_path: Path,
) -> OptimizerStageArtifact:
    try:
        lowered_ref = lowered_index_path.relative_to(package_dir).as_posix()
    except ValueError as error:
        raise ModelCompileError(
            "optimizer exact baseline must be inside the compiled model package"
        ) from error
    baseline_digest = contract_digest(lowered_index)
    session = OptimizationSession.create(package_id, baseline_digest)
    optimizer_dir = package_dir / OPTIMIZER_STAGE_DIR
    scope_catalog = write_optimization_scope_catalog(
        package_id=package_id,
        package_dir=package_dir,
        optimizer_dir=optimizer_dir,
        lowered_index=lowered_index,
        lowered_index_ref=lowered_ref,
    )
    exact_baseline = {
        "artifact_ref": lowered_ref,
        "contract_digest": baseline_digest,
        "mutable": False,
    }
    implementation_registry = create_empty_implementation_registry(
        package_id=package_id,
        exact_baseline=exact_baseline,
    )
    implementation_registry_path = (
        optimizer_dir / OPTIMIZER_IMPLEMENTATION_REGISTRY_FILE
    )
    write_json(
        implementation_registry_path,
        implementation_registry.to_json(),
    )
    document = {
        "schema": OPTIMIZER_STAGE_SCHEMA,
        "stage": "behavioral_representation_optimization",
        "compiler_position": {
            "after": "exact_semantic_lowering",
            "before": "physical_package_publication",
        },
        "status": "exact_baseline_retained",
        "exact_baseline": exact_baseline,
        "scope_catalog": {
            "artifact_ref": scope_catalog.package_reference(package_dir),
            "contract_digest": scope_catalog.digest,
            "scope_count": scope_catalog.document["summary"]["scope_count"],
            "rejected_scope_count": scope_catalog.document["summary"][
                "rejected_scope_count"
            ],
        },
        "session": session.to_json(),
        "implementation_registry": {
            "artifact_ref": (
                f"{OPTIMIZER_STAGE_DIR}/"
                f"{OPTIMIZER_IMPLEMENTATION_REGISTRY_FILE}"
            ),
            "contract_digest": contract_digest(
                implementation_registry.to_json()
            ),
            "implementation_count": 0,
        },
        "contract_schemas": list(OPTIMIZER_CONTRACT_SCHEMAS),
    }
    validate_optimizer_stage(document, package_dir=package_dir)
    path = package_dir / OPTIMIZER_STAGE_DIR / OPTIMIZER_STAGE_FILE
    write_json(path, document)
    return OptimizerStageArtifact(path=path, document=document)


def validate_optimizer_stage(document: Json, *, package_dir: Path) -> None:
    required = {
        "schema",
        "stage",
        "compiler_position",
        "status",
        "exact_baseline",
        "scope_catalog",
        "session",
        "implementation_registry",
        "contract_schemas",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise ModelCompileError(
            "compiled representation-optimizer stage contract has invalid fields"
        )
    if (
        document["schema"] != OPTIMIZER_STAGE_SCHEMA
        or document["stage"] != "behavioral_representation_optimization"
        or document["compiler_position"]
        != {
            "after": "exact_semantic_lowering",
            "before": "physical_package_publication",
        }
        or document["status"] not in {
            "exact_baseline_retained",
            "optimized",
        }
        or document["contract_schemas"] != list(OPTIMIZER_CONTRACT_SCHEMAS)
    ):
        raise ModelCompileError(
            "compiled representation-optimizer stage contract is unsupported"
        )
    baseline = document["exact_baseline"]
    if (
        not isinstance(baseline, dict)
        or set(baseline) != {"artifact_ref", "contract_digest", "mutable"}
        or baseline.get("mutable") is not False
    ):
        raise ModelCompileError(
            "representation optimizer has no immutable exact baseline contract"
        )
    baseline_path = _package_artifact_path(
        package_dir,
        baseline.get("artifact_ref"),
        "representation optimizer exact baseline",
    )
    if not baseline_path.is_file():
        raise ModelCompileError(
            f"representation optimizer exact baseline is missing: {baseline_path}"
        )
    baseline_document = read_json(baseline_path)
    if contract_digest(baseline_document) != baseline.get("contract_digest"):
        raise ModelCompileError(
            "representation optimizer exact baseline digest does not match"
        )
    scope_catalog = _require_object(
        document["scope_catalog"],
        "optimization scope catalog",
    )
    if set(scope_catalog) != {
        "artifact_ref",
        "contract_digest",
        "scope_count",
        "rejected_scope_count",
    }:
        raise ModelCompileError(
            "compiled representation optimizer scope catalog reference is invalid"
        )
    scope_catalog_path = _package_artifact_path(
        package_dir,
        scope_catalog.get("artifact_ref"),
        "optimization scope catalog",
    )
    if not scope_catalog_path.is_file():
        raise ModelCompileError(
            f"optimization scope catalog is missing: {scope_catalog_path}"
        )
    scope_catalog_document = load_optimization_scope_catalog(scope_catalog_path)
    if contract_digest(scope_catalog_document) != scope_catalog.get(
        "contract_digest"
    ):
        raise ModelCompileError(
            "optimization scope catalog digest does not match"
        )
    summary = scope_catalog_document["summary"]
    if (
        scope_catalog.get("scope_count") != summary["scope_count"]
        or scope_catalog.get("rejected_scope_count")
        != summary["rejected_scope_count"]
    ):
        raise ModelCompileError(
            "optimization scope catalog counts do not match"
        )
    session = OptimizationSession.from_json(
        _require_object(document["session"], "optimizer session")
    )
    if session.exact_baseline_digest != baseline["contract_digest"]:
        raise ModelCompileError(
            "optimizer session does not reference the immutable exact baseline"
        )
    registry_reference = _require_object(
        document["implementation_registry"],
        "implementation registry reference",
    )
    if set(registry_reference) != {
        "artifact_ref",
        "contract_digest",
        "implementation_count",
    }:
        raise ModelCompileError(
            "compiled implementation registry reference is invalid"
        )
    registry_path = _package_artifact_path(
        package_dir,
        registry_reference.get("artifact_ref"),
        "implementation registry",
    )
    if not registry_path.is_file():
        raise ModelCompileError(
            f"compiled implementation registry is missing: {registry_path}"
        )
    registry = ImplementationRegistry.from_json(read_json(registry_path))
    registry_document = registry.to_json()
    if (
        registry_document["package_id"] != session.package_id
        or registry_document["exact_baseline"] != baseline
        or registry_reference.get("contract_digest")
        != contract_digest(registry_document)
        or registry_reference.get("implementation_count")
        != len(registry.implementations)
    ):
        raise ModelCompileError(
            "compiled implementation registry does not match optimizer stage"
        )
    validate_published_implementation_registry(
        package_dir,
        registry,
        scope_catalog=scope_catalog_document,
    )
    if document["status"] == "exact_baseline_retained" and (
        session.candidates or registry.implementations
    ):
        raise ModelCompileError(
            "an unoptimized stage cannot declare representation candidates"
        )
    if document["status"] == "optimized":
        if any(
            candidate.state.value != "published"
            for candidate in session.candidates
        ):
            raise ModelCompileError(
                "optimized package sessions may contain only published candidates"
            )
        published_candidates = {
            candidate.candidate_id
            for candidate in session.candidates
            if candidate.state.value == "published"
        }
        registry_candidates = {
            str(implementation["candidate_id"])
            for implementation in registry.implementations
        }
        if (
            not registry_candidates
            or registry_candidates != published_candidates
        ):
            raise ModelCompileError(
                "optimized stage registry and published candidate lifecycle differ"
            )
        for candidate in session.candidates:
            for event in candidate.history:
                for evidence_ref in event["evidence_refs"]:
                    evidence_path = _package_artifact_path(
                        package_dir,
                        evidence_ref,
                        "published candidate lifecycle evidence",
                    )
                    if not evidence_path.is_file():
                        raise ModelCompileError(
                            "published candidate lifecycle evidence is missing"
                        )


def load_optimizer_stage(path: Path, *, package_dir: Path) -> Json:
    document = read_json(path)
    validate_optimizer_stage(document, package_dir=package_dir)
    return document


def _package_artifact_path(package_dir: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ModelCompileError(f"compiled package has no {label} path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ModelCompileError(
            f"compiled package {label} path must stay inside the package"
        )
    return package_dir / relative


def _require_object(value: Any, label: str) -> Json:
    if not isinstance(value, dict):
        raise ModelCompileError(f"{label} must be an object")
    return value
