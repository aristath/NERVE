from __future__ import annotations


OVERLAY_PATH = "overlays/component.json"
DECODE_SHADER_PATH = "kernels/head_norm_rope_embedded_parameters.spv"
PROOF_PATH = "proofs/embedded_parameter_program_equivalence.json"
COMPONENT_FIXTURE_PATH = "fixtures/head_norm_inputs.json"
CONVERSATION_FIXTURE_PATH = "fixtures/conversation.json"
MODEL_LIMITS_PATH = "fixtures/model_limits.json"


def embedded_artifact_paths() -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                COMPONENT_FIXTURE_PATH,
                CONVERSATION_FIXTURE_PATH,
                DECODE_SHADER_PATH,
                MODEL_LIMITS_PATH,
                OVERLAY_PATH,
                PROOF_PATH,
            )
        )
    )
