from __future__ import annotations

from nerve.compilation import Json
from nerve.representation_optimizer.providers.codebook.artifacts import (
    conversation_fixture,
    model_limits_fixture,
    product_conversation_fixture,
)


WEIGHT_PATH = "parameters/output_projection.safetensors"
SCALE_PATH = "parameters/output_projection_scales.safetensors"
TENSOR_FRAGMENT_PATH = "parameters/tensors.json"
OVERLAY_PATH = "overlays/output_transducer.json"
DECODE_SHADER_PATH = "kernels/output_projection.spv"
BATCH_SHADER_PATH = "kernels/output_projection_batch.spv"
ERROR_REPORT_PATH = "proofs/quantization_error.json"
COMPONENT_FIXTURE_PATH = "fixtures/output_projection_inputs.json"
CONVERSATION_FIXTURE_PATH = "fixtures/conversation.json"
PRODUCT_CONVERSATION_FIXTURE_PATH = "fixtures/product_conversation.json"
MODEL_LIMITS_PATH = "fixtures/model_limits.json"


def artifact_paths() -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                BATCH_SHADER_PATH,
                COMPONENT_FIXTURE_PATH,
                CONVERSATION_FIXTURE_PATH,
                DECODE_SHADER_PATH,
                ERROR_REPORT_PATH,
                MODEL_LIMITS_PATH,
                OVERLAY_PATH,
                PRODUCT_CONVERSATION_FIXTURE_PATH,
                SCALE_PATH,
                TENSOR_FRAGMENT_PATH,
                WEIGHT_PATH,
            )
        )
    )


def component_fixture(
    *,
    component_id: str,
    physical_node_id: str,
    hidden_size: int,
    vocabulary_size: int,
) -> Json:
    return {
        "schema": "nerve.optimizer.output_projection_fixture.v1",
        "component_id": component_id,
        "physical_node_id": physical_node_id,
        "input_shape": [hidden_size],
        "output_shape": [vocabulary_size],
        "input_dtype": "BF16",
        "output_dtype": "F32",
        "generator": {
            "kind": "deterministic_bounded_f32_to_bf16",
            "minimum": -4.0,
            "maximum": 4.0,
        },
    }


__all__ = [
    "BATCH_SHADER_PATH",
    "COMPONENT_FIXTURE_PATH",
    "CONVERSATION_FIXTURE_PATH",
    "DECODE_SHADER_PATH",
    "ERROR_REPORT_PATH",
    "MODEL_LIMITS_PATH",
    "OVERLAY_PATH",
    "PRODUCT_CONVERSATION_FIXTURE_PATH",
    "SCALE_PATH",
    "TENSOR_FRAGMENT_PATH",
    "WEIGHT_PATH",
    "artifact_paths",
    "component_fixture",
    "conversation_fixture",
    "model_limits_fixture",
    "product_conversation_fixture",
]
