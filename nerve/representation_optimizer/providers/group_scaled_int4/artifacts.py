from __future__ import annotations

from pathlib import PurePosixPath

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import stable_contract_id
from nerve.representation_optimizer.providers.codebook.artifacts import (
    conversation_fixture,
    model_limits_fixture,
    product_conversation_fixture,
)


REPORT_PATH = "proofs/group_scaled_int4_quantization.json"
COMPONENT_FIXTURE_PATH = "fixtures/group_scaled_int4_component.json"
CONVERSATION_FIXTURE_PATH = "fixtures/conversation.json"
PRODUCT_CONVERSATION_FIXTURE_PATH = "fixtures/product_conversation.json"
MODEL_LIMITS_PATH = "fixtures/model_limits.json"
TENSOR_FRAGMENT_PATH = "parameters/tensors.json"


def region_token(component_id: str, node_id: str) -> str:
    return stable_contract_id("group_scaled_int4_region", component_id, node_id)


def weight_artifact_path(component_id: str, node_id: str) -> str:
    token = region_token(component_id, node_id)
    return f"parameters/group_scaled_int4/{token}.weight.safetensors"


def scale_artifact_path(component_id: str, node_id: str) -> str:
    token = region_token(component_id, node_id)
    return f"parameters/group_scaled_int4/{token}.scales.safetensors"


def component_overlay_path(component_id: str, node_id: str) -> str:
    return f"overlays/{region_token(component_id, node_id)}.json"


def kernel_artifact_path(shader_file: str) -> str:
    path = PurePosixPath(shader_file)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.suffix != ".comp"
        or len(path.parts) != 1
    ):
        raise ModelCompileError(
            f"unsupported group-scaled INT4 shader path {shader_file!r}"
        )
    return f"kernels/group_scaled_int4/{path.with_suffix('.spv').name}"


def component_fixture(
    *,
    component_id: str,
    node_id: str,
    input_features: int,
    output_features: int,
) -> Json:
    return {
        "schema": "nerve.optimizer.group_scaled_int4_fixture.v1",
        "component_id": component_id,
        "physical_node_id": node_id,
        "input_shape": [input_features],
        "output_shape": [output_features],
        "input_dtype": "BF16",
        "output_dtype": "BF16",
        "generator": {
            "kind": "deterministic_bounded_f32_to_bf16",
            "minimum": -4.0,
            "maximum": 4.0,
        },
        "edge_cases": [
            "zeros",
            "finite_extrema",
            "alternating_signs",
            "deterministic_random",
        ],
    }


__all__ = [
    "COMPONENT_FIXTURE_PATH",
    "CONVERSATION_FIXTURE_PATH",
    "MODEL_LIMITS_PATH",
    "PRODUCT_CONVERSATION_FIXTURE_PATH",
    "REPORT_PATH",
    "TENSOR_FRAGMENT_PATH",
    "component_fixture",
    "component_overlay_path",
    "conversation_fixture",
    "kernel_artifact_path",
    "model_limits_fixture",
    "product_conversation_fixture",
    "region_token",
    "scale_artifact_path",
    "weight_artifact_path",
]
