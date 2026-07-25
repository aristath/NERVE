from __future__ import annotations

import sys

import pytest

from nerve.conversation_gate import (
    CANONICAL_OUTPUT_TOKEN_ALLOWANCE,
    MEASURED_PROMPTS,
    WARMUP_PROMPT,
    ConversationGateError,
    ConversationTurn,
    canonical_runtime_command,
    parse_conversation_transcript,
    repeated_suffix,
    run_resident_conversation,
    validate_conversation_turns,
)


def _response(answer: str) -> str:
    return f"Reasoning through the request carefully.\n</think>\n\n{answer}"


def _turn(prompt: str, answer: str, decode: float = 25.0) -> ConversationTurn:
    return ConversationTurn(
        prompt=prompt,
        response=_response(answer),
        stats={
            "decode_tokens_per_second": decode,
            "prefill_tokens_per_second": 100.0,
        },
    )


def _valid_turns() -> list[ConversationTurn]:
    return [
        _turn(MEASURED_PROMPTS[0], "I am a language model."),
        _turn(MEASURED_PROMPTS[1], "The capital is Athens."),
        _turn(MEASURED_PROMPTS[2], "Corinth refers to cities in Greece and elsewhere."),
        _turn(MEASURED_PROMPTS[3], "My knowledge cutoff is not available."),
        _turn(MEASURED_PROMPTS[4], "You asked about Greece."),
    ]


def test_transcript_parser_requires_all_completed_resident_turns() -> None:
    sections = ["ready\n"]
    for prompt, answer in (
        (WARMUP_PROMPT, "Hello."),
        *zip(
            MEASURED_PROMPTS,
            (
                "I am a language model.",
                "Athens.",
                "There are cities called Corinth.",
                "My cutoff is unknown.",
                "Greece.",
            ),
            strict=True,
        ),
    ):
        sections.append(
            "you> llm> "
            f"{_response(answer)}\n"
            "stats:\n"
            "  setup_ms=0.000\n"
            "  prefill_tokens_per_second=100.000\n"
            "  decode_tokens_per_second=25.000\n"
            "execution:\n"
            "  resident_sequence_queue_submits=1\n"
        )
    sections.append("you> ")

    turns = parse_conversation_transcript("".join(sections))

    assert [turn.prompt for turn in turns] == [WARMUP_PROMPT, *MEASURED_PROMPTS]
    assert turns[-1].stats["decode_tokens_per_second"] == 25.0


def test_transcript_parser_does_not_treat_quoted_user_prompt_as_a_turn() -> None:
    sections = ["ready\n"]
    for answer in (
        "I can quote the marker you> without opening a turn.",
        "I am a language model.",
        "Athens.",
        "There are cities called Corinth.",
        "My cutoff is unknown.",
        "Greece.",
    ):
        sections.append(
            "you> llm> "
            f"{_response(answer)}\n"
            "stats:\n"
            "  prefill_tokens_per_second=100.000\n"
            "  decode_tokens_per_second=25.000\n"
            "execution:\n"
        )
    sections.append("you> ")

    turns = parse_conversation_transcript("".join(sections))

    assert len(turns) == 6
    assert "quote the marker you>" in turns[0].response


def test_transcript_parser_rejects_a_missing_turn_even_if_prior_text_is_valid() -> None:
    transcript = (
        "you> llm> Reasoning</think> Hello\n"
        "stats:\n"
        "  decode_tokens_per_second=25.0\n"
        "execution:\n"
    )

    with pytest.raises(ConversationGateError, match="1 completed turn"):
        parse_conversation_transcript(transcript)


@pytest.mark.parametrize(
    "text",
    (
        "Output matches Done Proceed " * 8,
        "😊" * 96,
        ("Corinth Greece city list continues " * 6).strip(),
    ),
)
def test_repeated_suffix_catches_text_token_and_unicode_loops(text: str) -> None:
    assert repeated_suffix(text) is not None


def test_repeated_suffix_allows_long_nonrepeating_reasoning() -> None:
    text = " ".join(f"distinct-step-{index}" for index in range(1_000))
    assert repeated_suffix(text) is None


def test_gate_rejects_malformed_thinking_boundary() -> None:
    turns = _valid_turns()
    turns[0] = ConversationTurn(
        prompt=turns[0].prompt,
        response="Reasoning without a closing boundary.",
        stats=turns[0].stats,
    )

    with pytest.raises(ConversationGateError, match="exactly one"):
        validate_conversation_turns(
            turns,
            require_thinking=True,
            minimum_decode_tokens_per_second=20.0,
        )


def test_gate_rejects_turn_contamination_instead_of_accepting_meaningful_text() -> None:
    turns = _valid_turns()
    turns[2] = _turn(MEASURED_PROMPTS[2], "The capital of Greece is Athens.")

    with pytest.raises(ConversationGateError, match="Corinth"):
        validate_conversation_turns(
            turns,
            require_thinking=True,
            minimum_decode_tokens_per_second=20.0,
        )


def test_gate_rejects_repeated_final_answer() -> None:
    turns = _valid_turns()
    turns[3] = _turn(
        MEASURED_PROMPTS[3],
        "Output matches Done Proceed " * 8,
    )

    with pytest.raises(ConversationGateError, match="repeated segment"):
        validate_conversation_turns(
            turns,
            require_thinking=True,
            minimum_decode_tokens_per_second=20.0,
        )


def test_gate_averages_all_five_measured_turns_and_enforces_floor() -> None:
    turns = [
        _turn(turn.prompt, turn.response.rsplit("</think>", 1)[-1], decode=20.0)
        for turn in _valid_turns()
    ]
    turns[0] = _turn(MEASURED_PROMPTS[0], "I am a language model.", decode=15.0)

    with pytest.raises(ConversationGateError, match="19.000 tok/s"):
        validate_conversation_turns(
            turns,
            require_thinking=True,
            minimum_decode_tokens_per_second=20.0,
        )


def test_gate_accepts_a_complete_correct_conversation() -> None:
    mean_decode, mean_prefill = validate_conversation_turns(
        _valid_turns(),
        require_thinking=True,
        minimum_decode_tokens_per_second=20.0,
    )

    assert mean_decode == 25.0
    assert mean_prefill == 100.0


def test_runtime_command_uses_normal_chat_and_canonical_output_allowance() -> None:
    command = canonical_runtime_command(
        ["nerve-runtime", "--package", "model.json", "--chat"],
        seed=7,
    )

    assert command[-4:] == [
        "--max-new-tokens",
        str(CANONICAL_OUTPUT_TOKEN_ALLOWANCE),
        "--seed",
        "7",
    ]


def test_runtime_command_replaces_seed_without_duplicating_it() -> None:
    command = canonical_runtime_command(
        [
            "nerve-runtime",
            "--package",
            "model.json",
            "--chat",
            "--max-new-tokens",
            str(CANONICAL_OUTPUT_TOKEN_ALLOWANCE),
            "--seed",
            "1",
        ],
        seed=2,
    )

    assert command.count("--seed") == 1
    assert command[-2:] == ["--seed", "2"]


def test_runtime_command_rejects_noncanonical_output_limit() -> None:
    with pytest.raises(ConversationGateError, match="requires --max-new-tokens 65536"):
        canonical_runtime_command(
            [
                "nerve-runtime",
                "--package",
                "model.json",
                "--chat",
                "--max-new-tokens",
                "512",
            ],
            seed=0,
        )


def test_resident_runner_waits_for_each_completed_turn_before_sending_next(
    tmp_path,
) -> None:
    fake_runtime = tmp_path / "fake_runtime.py"
    fake_runtime.write_text(
        """
import sys

print("ready")
for turn in range(7):
    print("you> ", end="", flush=True)
    prompt = sys.stdin.readline().rstrip("\\n")
    if prompt == "/exit":
        break
    quoted = " quoted you> marker" if turn == 0 else ""
    print(f"llm> reasoning{quoted}</think> answer")
    print("stats:")
    print("  prefill_tokens_per_second=100.000")
    print("  decode_tokens_per_second=25.000")
    print("execution:")
    print("  resident_sequence_queue_submits=1", flush=True)
""".lstrip()
    )

    transcript, return_code = run_resident_conversation(
        [sys.executable, "-u", str(fake_runtime)]
    )

    assert return_code == 0
    assert len(parse_conversation_transcript(transcript)) == 6
