"""Representation-provider contracts and model-neutral provider registry."""

from nerve.representation_optimizer.providers.protocol import (
    RepresentationProvider,
)
from nerve.representation_optimizer.providers.registry import ProviderRegistry
from nerve.representation_optimizer.providers.source_artifacts import (
    PackageSourceArtifactResolver,
    SourceArtifact,
    SourceArtifactResolver,
    SourceTensorArtifact,
)
from nerve.representation_optimizer.providers.types import (
    EvidenceAssessment,
    MatchAssessment,
    ProviderCandidatePlan,
    ProviderIdentity,
    ProviderProblem,
    ProviderRegistryReport,
    StaticEstimate,
)

__all__ = [
    "EvidenceAssessment",
    "MatchAssessment",
    "ProviderCandidatePlan",
    "ProviderIdentity",
    "ProviderProblem",
    "ProviderRegistry",
    "PackageSourceArtifactResolver",
    "SourceArtifact",
    "SourceArtifactResolver",
    "SourceTensorArtifact",
    "ProviderRegistryReport",
    "RepresentationProvider",
    "StaticEstimate",
]
