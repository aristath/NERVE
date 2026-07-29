from __future__ import annotations

from types import SimpleNamespace

import pytest

from nerve.compilation import ModelCompileError
from nerve.representation_optimizer.validation.conversation_semantics import (
    assess_semantic_conversation,
    compare_semantic_conversations,
    validate_semantic_expectations,
)
from nerve.representation_optimizer.validation.runner import (
    _behavioral_rejection,
)


def _fixture() -> dict:
    return {
        "schema": "nerve.optimizer.validation_conversation.v1",
        "turns": [
            "what is the capital of Greece?",
            "Which country did I ask about?",
        ],
        "teacher_forced_assistant_turns": [
            "Athens.",
            "Greece.",
        ],
        "semantic_expectations": {
            "require_thinking": True,
            "forbid_repeated_suffix": True,
            "turns": [
                {
                    "required_terms": ["athens"],
                    "conversation_memory": False,
                },
                {
                    "required_terms": ["greece"],
                    "conversation_memory": True,
                },
            ],
        },
    }


def _trace(first: str, second: str) -> dict:
    return {
        "turns": [
            {
                "user": "what is the capital of Greece?",
                "assistant": f"reasoning</think>\n\n{first}",
            },
            {
                "user": "Which country did I ask about?",
                "assistant": f"reasoning</think>\n\n{second}",
            },
        ]
    }


def _request() -> dict:
    return {
        "check": {
            "metrics": [
                "conversation_memory",
                "semantic_consistency",
            ]
        }
    }


def test_semantic_comparison_accepts_distinct_correct_trajectories() -> None:
    comparison = compare_semantic_conversations(
        _request(),
        _fixture(),
        _trace(
            "Athens is the capital.",
            "You asked about Greece.",
        ),
        _trace(
            "The answer is Athens.",
            "The country was Greece.",
        ),
    )

    assert comparison["diagnostics"] == []
    assert comparison["metrics"] == [
        {
            "name": "conversation_memory",
            "reference_value": 1.0,
            "candidate_value": 1.0,
            "error": 0.0,
            "unit": "boolean_success",
        },
        {
            "name": "semantic_consistency",
            "reference_value": 1.0,
            "candidate_value": 1.0,
            "error": 0.0,
            "unit": "boolean_success",
        },
    ]


def test_semantic_comparison_rejects_candidate_memory_failure() -> None:
    comparison = compare_semantic_conversations(
        _request(),
        _fixture(),
        _trace(
            "Athens is the capital.",
            "You asked about Greece.",
        ),
        _trace(
            "The answer is Athens.",
            "I cannot recall the country.",
        ),
    )

    by_name = {
        metric["name"]: metric
        for metric in comparison["metrics"]
    }
    assert by_name["conversation_memory"]["error"] == 1.0
    assert by_name["semantic_consistency"]["error"] == 1.0
    assert comparison["diagnostics"] == [
        "candidate: conversation turn 1 is missing semantic terms ['greece']"
    ]


def test_semantic_assessment_rejects_repetition_and_malformed_thinking() -> None:
    repeated = "Athens reply " * 20
    assessment = assess_semantic_conversation(
        _fixture(),
        _trace(repeated, "The country was Greece."),
    )
    assert assessment.semantic_consistency is False
    assert any("repeated output" in item for item in assessment.diagnostics)

    malformed = assess_semantic_conversation(
        _fixture(),
        {
            "turns": [
                {
                    "user": "what is the capital of Greece?",
                    "assistant": "Athens.",
                },
                {
                    "user": "Which country did I ask about?",
                    "assistant": "reasoning</think>\n\nGreece.",
                },
            ]
        },
    )
    assert malformed.semantic_consistency is False
    assert any(
        "malformed reasoning boundaries" in item
        for item in malformed.diagnostics
    )


def test_semantic_expectations_require_memory_and_canonical_terms() -> None:
    fixture = _fixture()
    fixture["semantic_expectations"]["turns"][1][
        "conversation_memory"
    ] = False
    with pytest.raises(ModelCompileError, match="memory-recall"):
        validate_semantic_expectations(fixture, turn_count=2)

    fixture = _fixture()
    fixture["semantic_expectations"]["turns"][1]["required_terms"] = [
        "Greece"
    ]
    with pytest.raises(ModelCompileError, match="invalid terms"):
        validate_semantic_expectations(fixture, turn_count=2)


def test_semantic_comparison_policy_allows_distinct_valid_trajectories() -> None:
    check = {
        "check_id": "semantic-conversation",
        "comparison": {
            "output_mode": "fixture_semantics",
            "state_mode": "trajectory_local",
        },
        "horizon": {
            "completion_condition": "semantic_stop_or_allowance_per_turn",
        },
        "metrics": [
            "conversation_memory",
            "semantic_consistency",
        ],
    }
    observation = {
        "status": "completed",
        "seed": 1,
        "reference": {
            "output_digest": "reference-output",
            "state_digest": "reference-state",
            "horizon_completion": {
                "condition": "semantic_stop_or_allowance_per_turn",
                "satisfied": True,
            },
        },
        "candidate": {
            "output_digest": "candidate-output",
            "state_digest": "candidate-state",
            "horizon_completion": {
                "condition": "semantic_stop_or_allowance_per_turn",
                "satisfied": True,
            },
        },
        "metrics": [
            {
                "name": "conversation_memory",
                "reference_value": 1.0,
                "candidate_value": 1.0,
                "error": 0.0,
            },
            {
                "name": "semantic_consistency",
                "reference_value": 1.0,
                "candidate_value": 1.0,
                "error": 0.0,
            },
        ],
    }
    plan = SimpleNamespace(
        behavioral_contract={
            "mode": "exact",
            "error_contract": None,
        }
    )

    assert _behavioral_rejection(plan, check, observation) is None

    check["comparison"] = {
        "output_mode": "exact_digest",
        "state_mode": "exact_digest",
    }
    assert _behavioral_rejection(
        plan,
        check,
        observation,
    ) == (
        "exact candidate diverged in output_digest during "
        "semantic-conversation"
    )
