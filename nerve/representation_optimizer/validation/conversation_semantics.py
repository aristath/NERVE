from __future__ import annotations

import re
from dataclasses import dataclass

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import canonical_json_bytes


_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class ConversationAssessment:
    semantic_consistency: bool
    conversation_memory: bool
    diagnostics: tuple[str, ...]


def validate_semantic_expectations(
    fixture: Json,
    *,
    turn_count: int,
) -> Json:
    expectations = fixture.get("semantic_expectations")
    if not isinstance(expectations, dict) or set(expectations) != {
        "require_thinking",
        "forbid_repeated_suffix",
        "turns",
    }:
        raise ModelCompileError(
            "semantic conversation fixture has no complete expectation contract"
        )
    if (
        not isinstance(expectations["require_thinking"], bool)
        or not isinstance(expectations["forbid_repeated_suffix"], bool)
        or not isinstance(expectations["turns"], list)
        or len(expectations["turns"]) != turn_count
    ):
        raise ModelCompileError(
            "semantic conversation expectations do not match fixture turns"
        )
    memory_turn_count = 0
    normalized_turns = []
    for index, raw_turn in enumerate(expectations["turns"]):
        if not isinstance(raw_turn, dict) or set(raw_turn) != {
            "required_terms",
            "conversation_memory",
        }:
            raise ModelCompileError(
                f"semantic expectation turn {index} is invalid"
            )
        raw_terms = raw_turn["required_terms"]
        memory = raw_turn["conversation_memory"]
        if (
            not isinstance(raw_terms, list)
            or not isinstance(memory, bool)
            or any(
                not isinstance(term, str)
                or not term.strip()
                or term != term.casefold()
                for term in raw_terms
            )
            or raw_terms != sorted(set(raw_terms))
        ):
            raise ModelCompileError(
                f"semantic expectation turn {index} has invalid terms"
            )
        memory_turn_count += int(memory)
        normalized_turns.append(
            {
                "required_terms": list(raw_terms),
                "conversation_memory": memory,
            }
        )
    if memory_turn_count == 0:
        raise ModelCompileError(
            "semantic conversation expectations need a memory-recall turn"
        )
    return {
        "require_thinking": expectations["require_thinking"],
        "forbid_repeated_suffix": expectations[
            "forbid_repeated_suffix"
        ],
        "turns": normalized_turns,
    }


def assess_semantic_conversation(
    fixture: Json,
    trace: Json,
) -> ConversationAssessment:
    turns = fixture.get("turns")
    if not isinstance(turns, list):
        raise ModelCompileError("conversation fixture turns are invalid")
    expectations = validate_semantic_expectations(
        fixture,
        turn_count=len(turns),
    )
    observed_turns = trace.get("turns")
    if not isinstance(observed_turns, list):
        raise ModelCompileError("conversation trace has no turns")

    diagnostics = []
    semantic_consistency = len(observed_turns) == len(turns)
    conversation_memory = semantic_consistency
    if not semantic_consistency:
        diagnostics.append(
            "conversation trace completed "
            f"{len(observed_turns)} turn(s); expected {len(turns)}"
        )

    for index, (expected_user, expectation) in enumerate(
        zip(turns, expectations["turns"], strict=True)
    ):
        if index >= len(observed_turns):
            if expectation["conversation_memory"]:
                conversation_memory = False
            continue
        observed = observed_turns[index]
        if not isinstance(observed, dict):
            semantic_consistency = False
            conversation_memory = False
            diagnostics.append(f"conversation turn {index} is not an object")
            continue
        user = observed.get("user")
        assistant = observed.get("assistant")
        if user != expected_user or not isinstance(assistant, str) or not assistant.strip():
            semantic_consistency = False
            if expectation["conversation_memory"]:
                conversation_memory = False
            diagnostics.append(
                f"conversation turn {index} has invalid user or assistant content"
            )
            continue

        answer = _visible_answer(
            assistant,
            require_thinking=expectations["require_thinking"],
        )
        if answer is None:
            semantic_consistency = False
            if expectation["conversation_memory"]:
                conversation_memory = False
            diagnostics.append(
                f"conversation turn {index} has malformed reasoning boundaries"
            )
            continue
        if (
            expectations["forbid_repeated_suffix"]
            and repeated_suffix(assistant) is not None
        ):
            semantic_consistency = False
            if expectation["conversation_memory"]:
                conversation_memory = False
            diagnostics.append(
                f"conversation turn {index} ends in repeated output"
            )

        folded_answer = answer.casefold()
        missing = [
            term
            for term in expectation["required_terms"]
            if term not in folded_answer
        ]
        if missing:
            semantic_consistency = False
            if expectation["conversation_memory"]:
                conversation_memory = False
            diagnostics.append(
                f"conversation turn {index} is missing semantic terms {missing}"
            )

    return ConversationAssessment(
        semantic_consistency=semantic_consistency,
        conversation_memory=conversation_memory,
        diagnostics=tuple(diagnostics),
    )


def compare_semantic_conversations(
    request: Json,
    fixture: Json,
    reference_trace: Json,
    candidate_trace: Json,
) -> Json:
    reference = assess_semantic_conversation(fixture, reference_trace)
    candidate = assess_semantic_conversation(fixture, candidate_trace)
    values = {
        "semantic_consistency": (
            reference.semantic_consistency,
            candidate.semantic_consistency,
        ),
        "conversation_memory": (
            reference.conversation_memory,
            candidate.conversation_memory,
        ),
        "token_exact_match": (
            True,
            canonical_json_bytes(reference_trace)
            == canonical_json_bytes(candidate_trace),
        ),
    }
    metrics = []
    for name in request["check"]["metrics"]:
        try:
            reference_value, candidate_value = values[name]
        except KeyError as error:
            raise ModelCompileError(
                f"semantic conversation comparator cannot measure {name!r}"
            ) from error
        passed = reference_value and candidate_value
        metrics.append(
            {
                "name": name,
                "reference_value": 1.0 if reference_value else 0.0,
                "candidate_value": 1.0 if candidate_value else 0.0,
                "error": 0.0 if passed else 1.0,
                "unit": "boolean_success",
            }
        )
    diagnostics = [
        *(f"reference: {item}" for item in reference.diagnostics),
        *(f"candidate: {item}" for item in candidate.diagnostics),
    ]
    return {
        "metrics": metrics,
        "diagnostics": diagnostics,
    }


def repeated_suffix(text: str, minimum_repeats: int = 4) -> str | None:
    normalized = re.sub(r"\s+", " ", text).strip()
    for width in (8, 12, 16, 24, 32, 48, 64, 96, 128, 192):
        if len(normalized) < width * minimum_repeats:
            continue
        suffix = normalized[-width:]
        if suffix.strip() and normalized.endswith(suffix * minimum_repeats):
            return suffix

    words = _WORD.findall(normalized)
    for width in (*range(4, 17), 24, 32, 48, 64):
        if len(words) < width * minimum_repeats:
            continue
        suffix = words[-width:]
        if all(
            words[-width * repeat : -width * (repeat - 1)] == suffix
            for repeat in range(2, minimum_repeats + 1)
        ):
            return " ".join(suffix)
    return None


def _visible_answer(
    response: str,
    *,
    require_thinking: bool,
) -> str | None:
    closing_count = response.count("</think>")
    opening_count = response.count("<think>")
    if require_thinking:
        if closing_count == 1 and opening_count <= 1:
            answer = response.rsplit("</think>", 1)[1].strip()
        elif (
            closing_count == 0
            and opening_count == 0
            and response.startswith(("thought\n", "analysis\n"))
        ):
            answer = response.split("\n", 1)[1].strip()
        else:
            return None
    else:
        if closing_count > 1 or opening_count > 1:
            return None
        answer = response.rsplit("</think>", 1)[-1].strip()
    return answer or None
