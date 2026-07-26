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
