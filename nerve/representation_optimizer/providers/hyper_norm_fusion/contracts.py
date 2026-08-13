HETEROGENEOUS_COMPOSITE_ISLAND_DESCRIPTOR_ID = (
    "representation_descriptor_13ceee038d9b37f3c4e4cfba31c4ca21"
)
TARGET_LOWERING_SCHEMA = "nerve.optimizer.hyper_norm_fusion_vulkan_lowering.v1"
PROOF_SCHEMA = "nerve.optimizer.hyper_norm_fusion_proof.v1"
PROOF_VERIFIER_ID = "nerve.exact_hyper_norm_fusion.v1"
COMPONENT_FIXTURE_SCHEMA = "nerve.optimizer.hyper_norm_component_fixture.v1"

EXACT_FUSION_OBLIGATIONS = (
    "component_region_overlay_replaces_only_proven_source_records",
    "hyper_norm_transaction_preserves_exact_source_semantics",
)
