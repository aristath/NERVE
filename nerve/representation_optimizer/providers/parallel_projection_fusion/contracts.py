HETEROGENEOUS_COMPOSITE_ISLAND_DESCRIPTOR_ID = (
    "representation_descriptor_13ceee038d9b37f3c4e4cfba31c4ca21"
)
TARGET_LOWERING_SCHEMA = "nerve.optimizer.parallel_projection_fusion_vulkan_lowering.v1"
PROOF_SCHEMA = "nerve.optimizer.parallel_projection_fusion_proof.v1"
PROOF_VERIFIER_ID = "nerve.exact_parallel_projection_fusion.v1"
COMPONENT_FIXTURE_SCHEMA = "nerve.optimizer.parallel_projection_component_fixture.v1"

EXACT_FUSION_OBLIGATIONS = (
    "component_region_overlay_replaces_only_proven_source_records",
    "parallel_projection_preserves_each_branch_dot_product_order",
)
COMBINED_UPSTREAM_FUSION_OBLIGATION = (
    "combined_upstream_producer_preserves_hyper_norm_and_prequant_order"
)
SUPPORTED_FUSION_OBLIGATIONS = (
    *EXACT_FUSION_OBLIGATIONS,
    COMBINED_UPSTREAM_FUSION_OBLIGATION,
)
