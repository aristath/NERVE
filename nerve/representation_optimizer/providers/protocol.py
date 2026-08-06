from __future__ import annotations

from typing import Protocol

from nerve.compilation import Json
from nerve.representation_optimizer.providers.types import (
    EvidenceAssessment,
    MatchAssessment,
    ProviderContext,
    ProviderIdentity,
    StaticEstimate,
)


class RepresentationProvider(Protocol):
    """Complete boundary between generic optimization and one representation."""

    identity: ProviderIdentity
    descriptor_id: str

    def may_optimize_scope(
        self,
        scope: Json,
        source_contract: Json,
    ) -> bool:
        """Conservatively route static scope contracts before analysis.

        Returning false promises that this provider cannot produce a candidate
        for the scope regardless of what algebraic analysis would discover.
        Implementations must return true whenever the static contract alone is
        insufficient to decide.
        """

    def required_analyzer_ids(
        self,
        scope: Json,
        source_contract: Json,
    ) -> tuple[str, ...]:
        """Declare the generic structural evidence required for this scope.

        The result must be sorted, unique, and non-empty whenever
        ``may_optimize_scope`` returns true. Representation-specific exact
        proofs and behavioral validation remain separate obligations; this
        declaration prevents unrelated generic analyzers from becoming an
        accidental prerequisite.
        """

    def match_semantics(self, context: ProviderContext) -> MatchAssessment:
        """Decide whether the scope's responsibility can use this representation."""

    def match_structure(self, context: ProviderContext) -> MatchAssessment:
        """Decide whether structural evidence supports attempting this representation."""

    def analyze_evidence(
        self,
        context: ProviderContext,
    ) -> EvidenceAssessment:
        """Interpret source evidence and state the facts used for synthesis."""

    def synthesize_candidates(
        self,
        context: ProviderContext,
        evidence: EvidenceAssessment,
    ) -> tuple[Json, ...]:
        """Produce deterministic representation-candidate contracts."""

    def emit_representation_ir(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        """Emit backend-neutral physical representation IR."""

    def lower_for_target(
        self,
        context: ProviderContext,
        candidate: Json,
        representation_ir: Json,
    ) -> Json:
        """Lower representation IR for the context's hardware profile."""

    def estimate_static_cost(
        self,
        context: ProviderContext,
        candidate: Json,
        representation_ir: Json,
        target_lowering: Json,
    ) -> StaticEstimate:
        """Estimate feasibility, bytes, construction, and steady-state work."""

    def construction_requirements(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        """Return the shared candidate_build_plan.v1 construction contract."""

    def mount_requirements(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        """Describe runtime residency, setup, and compatibility requirements."""

    def proof_or_error_contract(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        """Return exact proof obligations or the approximation error contract."""

    def benchmark_workloads(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> tuple[Json, ...]:
        """Declare shared benchmark_workload.v1 matched execution regimes."""

    def validation_requirements(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        """Return shared validation_requirements.v1 with complete coverage."""
