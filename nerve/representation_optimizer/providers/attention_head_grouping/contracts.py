HETEROGENEOUS_COMPOSITE_ISLAND_DESCRIPTOR_ID = (
    "representation_descriptor_13ceee038d9b37f3c4e4cfba31c4ca21"
)
TARGET_LOWERING_SCHEMA = "nerve.optimizer.attention_head_grouping_vulkan_lowering.v1"
PROOF_SCHEMA = "nerve.optimizer.attention_head_grouping_proof.v1"
PROOF_VERIFIER_ID = "nerve.exact_attention_head_grouping.v1"

EXACT_HEAD_GROUPING_OBLIGATIONS = (
    "component_region_overlay_replaces_only_proven_source_records",
    "grouped_attention_preserves_each_head_reduction_order",
)
