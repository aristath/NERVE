from __future__ import annotations

from nerve.compilation import Json
from nerve.representation_optimizer.contracts import canonical_json_bytes
from nerve.representation_optimizer.providers.codebook.discovery import (
    HeadNormCodebookOpportunity,
)
from nerve.representation_optimizer.staging.contracts import staged_artifact_digest


CODEBOOK_TENSOR_PATH = "parameters/codebook.safetensors"
BRANCH_INDEX_PATHS = (
    "parameters/branch_0_indices.safetensors",
    "parameters/branch_1_indices.safetensors",
)
TENSOR_FRAGMENT_PATH = "parameters/tensors.json"
OVERLAY_PATH = "overlays/component.json"
DECODE_SHADER_PATH = "kernels/head_norm_rope_codebook_u8.spv"
PREFILL_SHADER_PATH = "kernels/head_norm_rope_codebook_u8_temporal.spv"
PROOF_PATH = "proofs/codebook_equivalence.json"
COMPONENT_FIXTURE_PATH = "fixtures/head_norm_inputs.json"
CONVERSATION_FIXTURE_PATH = "fixtures/conversation.json"
PRODUCT_CONVERSATION_FIXTURE_PATH = "fixtures/product_conversation.json"
MODEL_LIMITS_PATH = "fixtures/model_limits.json"


def candidate_tensor_names(
    opportunity: HeadNormCodebookOpportunity,
) -> tuple[str, str, str]:
    namespace = f"nerve.optimizer.codebook.{opportunity.scope_id}"
    return (
        f"{namespace}.branch_0_indices",
        f"{namespace}.branch_1_indices",
        f"{namespace}.entries",
    )


def artifact_paths() -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                *BRANCH_INDEX_PATHS,
                CODEBOOK_TENSOR_PATH,
                COMPONENT_FIXTURE_PATH,
                CONVERSATION_FIXTURE_PATH,
                DECODE_SHADER_PATH,
                MODEL_LIMITS_PATH,
                OVERLAY_PATH,
                PREFILL_SHADER_PATH,
                PRODUCT_CONVERSATION_FIXTURE_PATH,
                PROOF_PATH,
                TENSOR_FRAGMENT_PATH,
            )
        )
    )


def component_fixture(opportunity: HeadNormCodebookOpportunity) -> Json:
    return component_fixture_from_geometry(
        component_id=opportunity.component_id,
        physical_node_id=opportunity.physical_node_id,
        head_width=opportunity.head_width,
        branch_head_counts=tuple(
            int(branch.attrs["head_count"]) for branch in opportunity.branches
        ),
    )


def component_fixture_from_geometry(
    *,
    component_id: str,
    physical_node_id: str,
    head_width: int,
    branch_head_counts: tuple[int, int],
) -> Json:
    return {
        "schema": "nerve.optimizer.head_norm_fixture.v1",
        "component_id": component_id,
        "physical_node_id": physical_node_id,
        "branch_widths": [head_count * head_width for head_count in branch_head_counts],
        "dtype": "BF16",
        "generator": {
            "kind": "deterministic_bounded_f32_to_bf16",
            "minimum": -4.0,
            "maximum": 4.0,
        },
    }


def conversation_fixture() -> Json:
    return {
        "schema": "nerve.optimizer.validation_conversation.v1",
        "enable_thinking": True,
        "turns": [
            "hi",
            "Who are you?",
            "what is the capital of Greece?",
            'How many cities named "Corinth" are there?',
            "What is your knowledge cutoff date?",
            ("I asked you earlier for the capital of a country. Which country was it?"),
        ],
        "teacher_forced_assistant_turns": [
            "Hello!",
            "I am a language model.",
            "The capital of Greece is Athens.",
            (
                "Several places have been named Corinth, including the ancient "
                "and modern Greek cities and cities in the United States."
            ),
            "My knowledge cutoff depends on the model release.",
            "You asked about Greece.",
        ],
        "semantic_expectations": {
            "require_thinking": True,
            "forbid_repeated_suffix": True,
            "turns": [
                {
                    "required_terms": [],
                    "conversation_memory": False,
                },
                {
                    "required_terms": ["language model"],
                    "conversation_memory": False,
                },
                {
                    "required_terms": ["athens"],
                    "conversation_memory": False,
                },
                {
                    "required_terms": ["corinth"],
                    "conversation_memory": False,
                },
                {
                    "required_terms": ["cutoff"],
                    "conversation_memory": False,
                },
                {
                    "required_terms": ["greece"],
                    "conversation_memory": True,
                },
            ],
        },
    }


def product_conversation_fixture() -> Json:
    return {
        "schema": "nerve.optimizer.validation_conversation.v1",
        "enable_thinking": True,
        "turns": [
            "hi",
            "Who are you?",
            "what is the capital of Greece?",
            (
                "I asked you earlier for the capital of a country. "
                "Which country was it?"
            ),
        ],
        "teacher_forced_assistant_turns": [
            "Hello!",
            "I am a language model.",
            "The capital of Greece is Athens.",
            "You asked about Greece.",
        ],
    }


def model_limits_fixture(max_context_activations: int) -> Json:
    return {
        "schema": "nerve.optimizer.model_limits_fixture.v1",
        "max_context_tokens": max_context_activations,
        "max_output_tokens": 65_536,
    }


def fixture_reference(path: str, document: Json) -> Json:
    return {
        "path": path,
        "digest": staged_artifact_digest(json_payload(document)),
    }


def json_payload(document: Json) -> bytes:
    return canonical_json_bytes(document) + b"\n"
