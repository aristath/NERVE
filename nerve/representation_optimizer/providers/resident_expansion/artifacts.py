from __future__ import annotations

from pathlib import PurePosixPath

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import stable_contract_id
from nerve.representation_optimizer.providers.codebook.artifacts import (
    conversation_fixture,
    model_limits_fixture,
    product_conversation_fixture,
)


PROOF_PATH = "proofs/exact_expansion.json"
COMPONENT_FIXTURE_PATH = "fixtures/component.json"
CONVERSATION_FIXTURE_PATH = "fixtures/conversation.json"
PRODUCT_CONVERSATION_FIXTURE_PATH = "fixtures/product_conversation.json"
MODEL_LIMITS_PATH = "fixtures/model_limits.json"


def adaptive_shader_artifact_path(source_shader_path: str) -> str:
    path = PurePosixPath(source_shader_path)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.suffix != ".spv"
        or "_mxfp4_e2m1_" not in path.name
        or "_resident_fp8_e4m3_" in path.name
        or "_adaptive_fp8_e4m3_" in path.name
    ):
        raise ModelCompileError(
            f"unsupported compact MXFP4 shader path {source_shader_path!r}"
        )
    target_name = path.name.replace(
        "_mxfp4_e2m1_",
        "_mxfp4_e2m1_adaptive_fp8_e4m3_",
        1,
    )
    return f"kernels/{target_name}"


def component_overlay_path(component_id: str) -> str:
    return f"overlays/{stable_contract_id('resident_region', component_id)}.json"


def artifact_paths(
    shader_paths: tuple[str, ...],
    component_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                PROOF_PATH,
                COMPONENT_FIXTURE_PATH,
                CONVERSATION_FIXTURE_PATH,
                PRODUCT_CONVERSATION_FIXTURE_PATH,
                MODEL_LIMITS_PATH,
                *shader_paths,
                *(
                    component_overlay_path(component_id)
                    for component_id in component_ids
                ),
            }
        )
    )


def component_fixture(
    *,
    component_id: str,
    node_ids: tuple[str, str],
    hidden_size: int,
    intermediate_size: int,
    expert_count: int,
    experts_per_token: int,
) -> Json:
    return {
        "schema": ("nerve.optimizer.exact_resident_expert_component_fixture.v1"),
        "component_id": component_id,
        "node_ids": list(node_ids),
        "input": {
            "dtype": "BF16",
            "shape": [hidden_size],
            "generator": {
                "kind": "deterministic_bounded_f32_to_bf16",
                "minimum": -4.0,
                "maximum": 4.0,
            },
        },
        "routing": {
            "expert_count": expert_count,
            "experts_per_token": experts_per_token,
            "generator": "deterministic_unique_selector_sets",
            "edge_cases": [
                "first_experts",
                "last_experts",
                "alternating_extremes",
                "deterministic_random",
            ],
        },
        "intermediate_size": intermediate_size,
        "output": {"dtype": "BF16", "shape": [hidden_size]},
    }


__all__ = [
    "COMPONENT_FIXTURE_PATH",
    "CONVERSATION_FIXTURE_PATH",
    "MODEL_LIMITS_PATH",
    "PRODUCT_CONVERSATION_FIXTURE_PATH",
    "PROOF_PATH",
    "artifact_paths",
    "component_overlay_path",
    "component_fixture",
    "conversation_fixture",
    "model_limits_fixture",
    "product_conversation_fixture",
    "adaptive_shader_artifact_path",
]
