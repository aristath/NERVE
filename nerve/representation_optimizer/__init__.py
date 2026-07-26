"""Target-aware behavioral representation optimization.

The optimizer consumes exact semantic circuits and may publish additional
verified physical implementations.  It never changes the runtime graph or
chooses runtime placement.
"""

from nerve.representation_optimizer.contracts import (
    ALGEBRAIC_EVIDENCE_SCHEMA,
    BENCHMARK_RECORD_SCHEMA,
    CANDIDATE_CONSTRUCTION_SCHEMA,
    HARDWARE_PROCESS_PROFILE_SCHEMA,
    OPTIMIZATION_SCOPE_SCHEMA,
    PROMOTION_DECISION_SCHEMA,
    RELOWERING_REQUEST_SCHEMA,
    REPRESENTATION_CANDIDATE_SCHEMA,
    REPRESENTATION_DESCRIPTOR_SCHEMA,
    SOURCE_BEHAVIOR_CONTRACT_SCHEMA,
    VALIDATION_RECORD_SCHEMA,
    ContractDocument,
    ContractValidationError,
    canonical_json_bytes,
    contract_digest,
    representation_descriptor_id,
    stable_contract_id,
    validate_contract,
)
from nerve.representation_optimizer.descriptor_registry import (
    RepresentationDescriptorRegistry,
    load_builtin_representation_descriptors,
)
from nerve.representation_optimizer.lifecycle import (
    CANDIDATE_LIFECYCLE_SCHEMA,
    OPTIMIZATION_SESSION_SCHEMA,
    CandidateLifecycle,
    CandidateState,
    OptimizationSession,
)

__all__ = [
    "ALGEBRAIC_EVIDENCE_SCHEMA",
    "BENCHMARK_RECORD_SCHEMA",
    "CANDIDATE_CONSTRUCTION_SCHEMA",
    "CANDIDATE_LIFECYCLE_SCHEMA",
    "HARDWARE_PROCESS_PROFILE_SCHEMA",
    "OPTIMIZATION_SCOPE_SCHEMA",
    "OPTIMIZATION_SESSION_SCHEMA",
    "PROMOTION_DECISION_SCHEMA",
    "RELOWERING_REQUEST_SCHEMA",
    "REPRESENTATION_CANDIDATE_SCHEMA",
    "REPRESENTATION_DESCRIPTOR_SCHEMA",
    "SOURCE_BEHAVIOR_CONTRACT_SCHEMA",
    "VALIDATION_RECORD_SCHEMA",
    "CandidateLifecycle",
    "CandidateState",
    "ContractDocument",
    "ContractValidationError",
    "OptimizationSession",
    "RepresentationDescriptorRegistry",
    "canonical_json_bytes",
    "contract_digest",
    "load_builtin_representation_descriptors",
    "representation_descriptor_id",
    "stable_contract_id",
    "validate_contract",
]
