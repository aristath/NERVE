from __future__ import annotations

from pathlib import PurePosixPath

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import stable_contract_id
from nerve.representation_optimizer.providers.codebook.artifacts import (
    conversation_fixture,
    model_limits_fixture,
    product_conversation_fixture,
)


PROOF_PATH = "proofs/exact_hyper_norm_fusion.json"
COMPONENT_FIXTURE_PATH = "fixtures/hyper_norm_component.json"
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
            f"unsupported fused hyper/RMS shader path {shader_file!r}"
        )
    return f"kernels/hyper_norm/{path.with_suffix('.spv').name}"


def component_overlay_path(component_id: str) -> str:
    return f"overlays/{stable_contract_id('hyper_norm_region', component_id)}.json"


def component_fixture(
    *,
    component_id: str,
    terminal_node_id: str,
    hidden_size: int,
) -> Json:
    return {
        "schema": "nerve.optimizer.hyper_norm_component_fixture.v1",
        "component_id": component_id,
        "terminal_node_id": terminal_node_id,
        "input": {
            "dtype": "BF16",
            "shape": [4, hidden_size],
            "generator": {
                "kind": "deterministic_bounded_f32_to_bf16",
                "minimum": -4.0,
                "maximum": 4.0,
            },
        },
        "output": {"dtype": "BF16", "shape": [4, hidden_size]},
        "edge_cases": [
            "zeros",
            "finite_extrema",
            "alternating_signs",
            "deterministic_random",
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
