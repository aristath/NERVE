from __future__ import annotations

from nerve.compilation import ModelCompileError
from nerve.representation_optimizer.descriptor_registry import (
    RepresentationDescriptorRegistry,
    load_builtin_representation_descriptors,
)
from nerve.representation_optimizer.providers.codebook import (
    CodebookToolchainResolver,
    EmbeddedParameterProgramToolchainResolver,
    ExactEmbeddedHeadNormParameterProgramProvider,
    ExactHeadNormCodebookProvider,
)
from nerve.representation_optimizer.providers.output_fp8 import (
    BlockScaledOutputProjectionProvider,
    BlockScaledOutputToolchainResolver,
)
from nerve.representation_optimizer.providers.resident_expansion import (
    ExactResidentExpertExpansionProvider,
    ResidentExpansionToolchainResolver,
)
from nerve.representation_optimizer.providers.registry import ProviderRegistry
from nerve.representation_optimizer.providers.types import ProviderCandidatePlan


def load_builtin_provider_registry(
    descriptors: RepresentationDescriptorRegistry | None = None,
) -> ProviderRegistry:
    return ProviderRegistry.from_providers(
        descriptors=descriptors or load_builtin_representation_descriptors(),
        providers=(
            ExactEmbeddedHeadNormParameterProgramProvider(),
            ExactHeadNormCodebookProvider(),
            BlockScaledOutputProjectionProvider(),
            ExactResidentExpertExpansionProvider(),
        ),
    )


class BuiltinCandidateToolchainResolver:
    """Resolve built-in provider plans without model-family dispatch."""

    def __init__(self) -> None:
        self._resolvers = (
            CodebookToolchainResolver(),
            EmbeddedParameterProgramToolchainResolver(),
            BlockScaledOutputToolchainResolver(),
            ResidentExpansionToolchainResolver(),
        )

    def resolve(self, plan: ProviderCandidatePlan):
        failures = []
        for resolver in self._resolvers:
            try:
                return resolver.resolve(plan)
            except ModelCompileError as error:
                failures.append(str(error))
        raise ModelCompileError(
            f"no built-in toolchain accepts provider "
            f"{plan.provider.provider_id!r}: {'; '.join(failures)}"
        )
