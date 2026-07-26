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
    OPTIMIZATION_SCOPE_CATALOG_SCHEMA,
    OPTIMIZATION_SCOPE_SCHEMA,
    PROMOTION_DECISION_SCHEMA,
    RELOWERING_REQUEST_SCHEMA,
    REPRESENTATION_CANDIDATE_SCHEMA,
    REPRESENTATION_DESCRIPTOR_SCHEMA,
    SOURCE_BEHAVIOR_CONTRACT_SCHEMA,
    VALIDATION_RECORD_SCHEMA,
    ContractDocument,
    ContractValidationError,
    algebraic_evidence_id,
    canonical_json_bytes,
    contract_digest,
    optimization_scope_catalog_id,
    representation_candidate_equivalence_key,
    representation_candidate_id,
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
from nerve.representation_optimizer.representation_ir import (
    REPRESENTATION_GRAPH_SCHEMA,
    RepresentationGraphDocument,
    RepresentationGraphPlan,
    finalize_representation_graph,
    plan_representation_graph,
    representation_graph_id,
    validate_representation_graph,
)

__all__ = [
    "ALGEBRAIC_EVIDENCE_SCHEMA",
    "BENCHMARK_RECORD_SCHEMA",
    "CANDIDATE_CONSTRUCTION_SCHEMA",
    "CANDIDATE_LIFECYCLE_SCHEMA",
    "HARDWARE_PROCESS_PROFILE_SCHEMA",
    "OPTIMIZATION_SCOPE_CATALOG_SCHEMA",
    "OPTIMIZATION_SCOPE_SCHEMA",
    "OPTIMIZATION_SESSION_SCHEMA",
    "PROMOTION_DECISION_SCHEMA",
    "RELOWERING_REQUEST_SCHEMA",
    "REPRESENTATION_CANDIDATE_SCHEMA",
    "REPRESENTATION_DESCRIPTOR_SCHEMA",
    "REPRESENTATION_GRAPH_SCHEMA",
    "SOURCE_BEHAVIOR_CONTRACT_SCHEMA",
    "VALIDATION_RECORD_SCHEMA",
    "CandidateLifecycle",
    "CandidateState",
    "ContractDocument",
    "ContractValidationError",
    "OptimizationSession",
    "RepresentationDescriptorRegistry",
    "RepresentationGraphDocument",
    "RepresentationGraphPlan",
    "algebraic_evidence_id",
    "canonical_json_bytes",
    "contract_digest",
    "finalize_representation_graph",
    "load_builtin_representation_descriptors",
    "optimization_scope_catalog_id",
    "plan_representation_graph",
    "representation_candidate_equivalence_key",
    "representation_candidate_id",
    "representation_descriptor_id",
    "representation_graph_id",
    "stable_contract_id",
    "validate_contract",
    "validate_representation_graph",
]
