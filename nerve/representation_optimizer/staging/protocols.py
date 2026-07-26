from __future__ import annotations

from typing import Protocol

from nerve.representation_optimizer.staging.workspace import (
    CandidateConstructionContext,
)


class CandidateSemanticConstructor(Protocol):
    def construct_semantic_artifacts(
        self,
        context: CandidateConstructionContext,
    ) -> None:
        """Construct provider-specific parameters, state, topology, and programs."""


class CandidateOrdinaryRelowerer(Protocol):
    def run_ordinary_lowering(
        self,
        context: CandidateConstructionContext,
    ) -> None:
        """Lower the semantic representation graph using ordinary compiler passes."""


class CandidatePhysicalOptimizer(Protocol):
    def optimize_physical_artifacts(
        self,
        context: CandidateConstructionContext,
    ) -> None:
        """Run target-aware physical optimization after semantic re-lowering."""
