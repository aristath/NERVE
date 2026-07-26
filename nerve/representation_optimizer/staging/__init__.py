"""Isolated candidate-construction contracts.

The filesystem orchestrator is intentionally not imported here. Provider type
definitions depend on the immutable build-plan contract, while the orchestrator
later depends on provider plans.
"""

from nerve.representation_optimizer.staging.contracts import (
    CANDIDATE_BUILD_PLAN_SCHEMA,
    STAGED_ARTIFACT_DIGEST_SCHEMA,
    CandidateBuildPlan,
    staged_artifact_digest,
    staged_file_digest,
)

__all__ = [
    "CANDIDATE_BUILD_PLAN_SCHEMA",
    "STAGED_ARTIFACT_DIGEST_SCHEMA",
    "CandidateBuildPlan",
    "staged_artifact_digest",
    "staged_file_digest",
]
