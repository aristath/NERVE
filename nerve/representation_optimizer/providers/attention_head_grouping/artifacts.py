from __future__ import annotations

from pathlib import PurePosixPath

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import stable_contract_id
from nerve.representation_optimizer.providers.codebook.artifacts import (
    conversation_fixture,
    model_limits_fixture,
    product_conversation_fixture,
)


PROOF_PATH = "proofs/exact_attention_head_grouping.json"
COMPONENT_FIXTURE_PATH = "fixtures/attention_component.json"
CONVERSATION_FIXTURE_PATH = "fixtures/conversation.json"
PRODUCT_CONVERSATION_FIXTURE_PATH = "fixtures/product_conversation.json"
MODEL_LIMITS_PATH = "fixtures/model_limits.json"


def kernel_artifact_path(shader_file: str) -> str:
    path = PurePosixPath(shader_file)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.suffix != ".comp"
        or len(path.parts) != 1
    ):
        raise ModelCompileError(
            f"unsupported grouped-attention shader path {shader_file!r}"
        )
    return f"kernels/attention_head_grouping/{path.with_suffix('.spv').name}"


def component_overlay_path(component_id: str) -> str:
    return (
        "overlays/"
        f"{stable_contract_id('attention_head_grouping_region', component_id)}.json"
    )


def component_fixture(
    *,
    component_id: str,
    physical_node_id: str,
    query_heads: int,
    head_width: int,
    local_window: int,
    max_compressed_indices: int,
) -> Json:
    inputs: list[Json] = [
        {
            "name": "query",
            "dtype": "BF16",
            "shape": [query_heads, head_width],
        },
        {
            "name": "local_state",
            "dtype": "BF16",
            "shape": [local_window, head_width],
        },
    ]
    if max_compressed_indices:
        inputs.extend(
            (
                {
                    "name": "compressed_state",
                    "dtype": "BF16",
                    "shape": [max_compressed_indices, head_width],
                },
                {
                    "name": "compressed_indices",
                    "dtype": "U32",
                    "shape": [max_compressed_indices],
                },
            )
        )
    return {
        "schema": "nerve.optimizer.attention_component_fixture.v1",
        "component_id": component_id,
        "physical_node_id": physical_node_id,
        "inputs": inputs,
        "parameters": [
            {
                "name": "attention_sinks",
                "dtype": "F32",
                "shape": [query_heads],
            }
        ],
        "output": {
            "dtype": "BF16",
            "shape": [query_heads, head_width],
        },
        "generator": {
            "kind": "deterministic_bounded_f32_to_declared_dtype",
            "minimum": -2.0,
            "maximum": 2.0,
        },
        "edge_cases": [
            "empty_history",
            "partial_local_window",
            "wrapped_local_window",
            "full_compressed_history",
            "finite_extrema",
        ],
    }


def artifact_paths(
    *,
    overlay_paths: tuple[str, ...],
    kernel_paths: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                PROOF_PATH,
                COMPONENT_FIXTURE_PATH,
                CONVERSATION_FIXTURE_PATH,
                PRODUCT_CONVERSATION_FIXTURE_PATH,
                MODEL_LIMITS_PATH,
                *overlay_paths,
                *kernel_paths,
            }
        )
    )


__all__ = [
    "COMPONENT_FIXTURE_PATH",
    "CONVERSATION_FIXTURE_PATH",
    "MODEL_LIMITS_PATH",
    "PRODUCT_CONVERSATION_FIXTURE_PATH",
    "PROOF_PATH",
    "artifact_paths",
    "component_fixture",
    "component_overlay_path",
    "conversation_fixture",
    "kernel_artifact_path",
    "model_limits_fixture",
    "product_conversation_fixture",
]
