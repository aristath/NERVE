from __future__ import annotations

import sys

import pytest

from nerve.conversation_gate import (
    CANONICAL_CONVERSATION_PROMPTS,
    CANONICAL_OUTPUT_TOKEN_ALLOWANCE,
    MEASURED_PROMPTS,
    WARMUP_PROMPT,
    ConversationGateError,
    ConversationTurn,
    canonical_runtime_command,
    parse_conversation_transcript,
    repeated_suffix,
    run_conversation_gate,
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


def test_transcript_parser_requires_every_discarded_and_measured_set() -> None:
    sections = ["ready\n"]
    for _ in range(2):
        for prompt in CANONICAL_CONVERSATION_PROMPTS:
            sections.append(
                "you> llm> "
                f"{_response(f'answer for {prompt}')}\n"
                "stats:\n"
                "  prefill_tokens_per_second=100.000\n"
                "  decode_tokens_per_second=25.000\n"
                "execution:\n"
            )
    sections.append("you> ")

    turns = parse_conversation_transcript(
        "".join(sections), warmup_conversation_sets=1
    )

    assert [turn.prompt for turn in turns] == list(
        CANONICAL_CONVERSATION_PROMPTS * 2
    )


def test_transcript_parser_extracts_cumulative_residency_counters() -> None:
    sections = ["ready\n"]
    for index, prompt in enumerate(CANONICAL_CONVERSATION_PROMPTS, start=1):
        sections.append(
            "you> llm> "
            f"{_response(f'answer for {prompt}')}\n"
            "stats:\n"
            "  prefill_tokens_per_second=100.000\n"
            "  decode_tokens_per_second=25.000\n"
            "execution:\n"
            "resource_residency:\n"
            "  policy=demand-paged physical_stores=3\n"
            f"  gpu_accesses(selections/resident_hits/misses)={index * 2}/{index}/{index}\n"
            f"  residency_requests(directory_hits/load_required/deduplicated/succeeded/failed/cancelled)={index}/{index}/0/{index}/0/0\n"
            "  residency_eviction(cycles/units/payload_bytes/device_bytes/reloads)=0/0/0/0/0\n"
            f"  memory_tiers(device_payload/host_visible_payload/device_capacity/host_visible_capacity)={index * 8}/{index * 2}/80/20\n"
            f"  transfers(reads/source_bytes/resident_bytes/uploaded_bytes/read_ms/derivation_ms/upload_ms/blocking_ms)={index}/{index * 10}/{index * 20}/{index * 20}/{index * 0.5}/{index * 0.125}/{index * 0.25}/{index * 0.75}\n"
            "determinism:\n"
        )
    sections.append("you> ")

    turns = parse_conversation_transcript("".join(sections))

    assert turns[-1].residency_policy == "demand-paged"
    assert turns[-1].residency_counters == {
        "gpu_accesses.selections": 12,
        "gpu_accesses.resident_hits": 6,
        "gpu_accesses.misses": 6,
        "residency_requests.directory_hits": 6,
        "residency_requests.load_required": 6,
        "residency_requests.deduplicated": 0,
        "residency_requests.succeeded": 6,
        "residency_requests.failed": 0,
        "residency_requests.cancelled": 0,
        "residency_eviction.cycles": 0,
        "residency_eviction.units": 0,
        "residency_eviction.payload_bytes": 0,
        "residency_eviction.device_bytes": 0,
        "residency_eviction.reloads": 0,
        "transfers.reads": 6,
        "transfers.source_bytes": 60,
        "transfers.resident_bytes": 120,
        "transfers.uploaded_bytes": 120,
        "transfers.read_ms": 3.0,
        "transfers.derivation_ms": 0.75,
        "transfers.upload_ms": 1.5,
        "transfers.blocking_ms": 4.5,
    }
    assert turns[-1].residency_gauges == {
        "memory_tiers.device_payload": 48,
        "memory_tiers.host_visible_payload": 12,
        "memory_tiers.device_capacity": 80,
        "memory_tiers.host_visible_capacity": 20,
    }


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


def test_repeated_suffix_catches_long_multiline_cycles() -> None:
    cycle = "\n".join(
        f"* There is a Corinth in region {index} with qualifier {index * 17}."
        for index in range(24)
    )

    repeated = repeated_suffix("Reasoning begins.\n" + "\n".join([cycle] * 5))

    assert repeated is not None
    assert "Corinth" in repeated


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

    with pytest.raises(ConversationGateError, match="must contain one"):
        validate_conversation_turns(
            turns,
            require_thinking=True,
            minimum_decode_tokens_per_second=20.0,
        )


@pytest.mark.parametrize("channel", ("thought", "analysis"))
def test_gate_accepts_decoded_reasoning_channels(channel: str) -> None:
    turns = _valid_turns()
    turns[0] = ConversationTurn(
        prompt=turns[0].prompt,
        response=f"{channel}\nReasoning through the request. I am a language model.",
        stats=turns[0].stats,
    )

    mean_decode, _ = validate_conversation_turns(
        turns,
        require_thinking=True,
        minimum_decode_tokens_per_second=20.0,
    )

    assert mean_decode == 25.0


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


def test_gate_requires_one_seed_per_gpu_residency_cycle(tmp_path) -> None:
    package = tmp_path / "package.json"
    package.write_text("{}")

    with pytest.raises(ConversationGateError, match="exactly one seed"):
        run_conversation_gate(
            ["nerve-runtime", "--package", str(package), "--chat"],
            seeds=(0, 1),
            minimum_decode_tokens_per_second=0.0,
            require_thinking=True,
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


def test_resident_runner_reports_bounded_runtime_failure_tail(tmp_path) -> None:
    fake_runtime = tmp_path / "failing_runtime.py"
    fake_runtime.write_text(
        """
import sys

print("ready")
print("you> ", end="", flush=True)
sys.stdin.readline()
print("x" * 5000)
print("precise terminal failure", flush=True)
raise SystemExit(7)
""".lstrip()
    )

    with pytest.raises(ConversationGateError) as caught:
        run_resident_conversation([sys.executable, "-u", str(fake_runtime)])

    message = str(caught.value)
    assert "runtime exited with status 7" in message
    assert "precise terminal failure" in message
    assert len(message) < 4_200


def test_resident_runner_stops_on_recoverable_turn_rejection(tmp_path) -> None:
    fake_runtime = tmp_path / "recoverable_turn_error.py"
    fake_runtime.write_text(
        """
import sys

print("ready")
print("you> ", end="", flush=True)
sys.stdin.readline()
print("llm> malformed generated protocol")
print("turn_error: generated assistant protocol validation failed before canonical commit: reserved token")
print("you> ", end="", flush=True)
sys.stdin.readline()
raise SystemExit(99)
""".lstrip()
    )

    with pytest.raises(
        ConversationGateError,
        match="runtime rejected a recoverable chat turn.*reserved token",
    ) as caught:
        run_resident_conversation([sys.executable, "-u", str(fake_runtime)])

    assert "malformed generated protocol" in caught.value.transcript
    assert "turn_error:" in caught.value.transcript


def test_gate_persists_complete_failed_runtime_transcript(tmp_path) -> None:
    package = tmp_path / "package.json"
    package.write_text("{}")
    fake_runtime = tmp_path / "failing_runtime.py"
    fake_runtime.write_text(
        """
import sys

print("ready")
print("you> ", end="", flush=True)
sys.stdin.readline()
print("failure-prefix")
print("x" * 5000)
print("precise terminal failure", flush=True)
raise SystemExit(7)
""".lstrip()
    )
    transcript_dir = tmp_path / "transcripts"

    with pytest.raises(ConversationGateError, match="runtime exited with status 7"):
        run_conversation_gate(
            [
                sys.executable,
                "-u",
                str(fake_runtime),
                "--package",
                str(package),
                "--chat",
            ],
            seeds=(19,),
            minimum_decode_tokens_per_second=0.0,
            require_thinking=True,
            transcript_dir=transcript_dir,
        )

    transcript = (transcript_dir / "conversation-seed-19-failed.log").read_text()
    assert "failure-prefix" in transcript
    assert "precise terminal failure" in transcript
    assert "x" * 5000 in transcript


def test_gate_discards_one_complete_resident_conversation_and_measures_the_next(
    tmp_path,
) -> None:
    package = tmp_path / "package.json"
    package.write_text("{}")
    fake_runtime = tmp_path / "two_set_runtime.py"
    fake_runtime.write_text(
        """
import sys

answers = (
    "Hello.",
    "I am a language model.",
    "The capital of Greece is Athens.",
    "There are several cities named Corinth.",
    "My knowledge cutoff is unavailable.",
    "The country was Greece.",
)
print("ready")
completed = 0
conversation_turn = 0
conversation_set = 0
while True:
    print("you> ", end="", flush=True)
    prompt = sys.stdin.readline().rstrip("\\n")
    if prompt == "/exit":
        break
    if prompt == "/new":
        if conversation_turn != len(answers):
            raise SystemExit(91)
        conversation_turn = 0
        conversation_set += 1
        print("session_reset: zeroed_state_buffers=17", flush=True)
        continue
    answer = answers[conversation_turn]
    conversation_turn += 1
    completed += 1
    misses = min(completed, len(answers))
    decode = 1.0 if conversation_set == 0 else 30.0
    print(f"llm> reasoning</think> {answer}")
    print("stats:")
    print("  prefill_tokens_per_second=100.000")
    print(f"  decode_tokens_per_second={decode:.3f}")
    print("execution:")
    print("resource_residency:")
    print("  policy=demand-paged physical_stores=3")
    print(f"  gpu_accesses(selections/resident_hits/misses)={completed}/{completed - misses}/{misses}")
    print(f"  residency_requests(directory_hits/load_required/deduplicated/succeeded/failed/cancelled)={misses}/{misses}/0/{misses}/0/0")
    print("  residency_eviction(cycles/units/payload_bytes/device_bytes/reloads)=0/0/0/0/0")
    print(f"  memory_tiers(device_payload/host_visible_payload/device_capacity/host_visible_capacity)={misses * 8}/{misses * 2}/80/20")
    print(f"  transfers(reads/source_bytes/resident_bytes/uploaded_bytes/read_ms/derivation_ms/upload_ms/blocking_ms)={misses}/{misses * 10}/{misses * 20}/{misses * 20}/{misses}.0/{misses / 2}/{misses}.0/{misses}.0", flush=True)
""".lstrip()
    )

    report = run_conversation_gate(
        [
            sys.executable,
            "-u",
            str(fake_runtime),
            "--package",
            str(package),
            "--chat",
        ],
        seeds=(0,),
        minimum_decode_tokens_per_second=20.0,
        require_thinking=True,
        warmup_conversation_sets=1,
    )

    run = report.runs[0]
    assert len(run.discarded_warmup_sets) == 1
    assert report.warmup_conversation_sets == 1
    assert run.discarded_warmup_sets[0].mean_decode_tokens_per_second == 1.0
    assert run.measured_set.mean_decode_tokens_per_second == 30.0
    assert run.measured_set.residency_policy == "demand-paged"
    assert run.measured_set.residency_delta is not None
    assert run.measured_set.residency_delta["gpu_accesses.selections"] == 6
    assert run.measured_set.residency_delta["gpu_accesses.misses"] == 0
    assert run.measured_set.residency_delta["transfers.blocking_ms"] == 0.0
    assert run.measured_set.residency_gauges_start == {
        "memory_tiers.device_payload": 48,
        "memory_tiers.host_visible_payload": 12,
        "memory_tiers.device_capacity": 80,
        "memory_tiers.host_visible_capacity": 20,
    }
    assert run.measured_set.residency_gauges_end == (
        run.measured_set.residency_gauges_start
    )


def test_resident_runner_rejects_a_runtime_that_does_not_acknowledge_session_reset(
    tmp_path,
) -> None:
    fake_runtime = tmp_path / "missing_reset_ack.py"
    fake_runtime.write_text(
        """
import sys

print("ready")
completed = 0
while True:
    print("you> ", end="", flush=True)
    prompt = sys.stdin.readline().rstrip("\\n")
    if prompt == "/exit":
        break
    if prompt == "/new":
        continue
    completed += 1
    print("llm> reasoning</think> answer")
    print("stats:")
    print("  prefill_tokens_per_second=100.000")
    print("  decode_tokens_per_second=25.000", flush=True)
""".lstrip()
    )

    with pytest.raises(
        ConversationGateError,
        match="did not acknowledge the required new-conversation reset",
    ):
        run_resident_conversation(
            [sys.executable, "-u", str(fake_runtime)],
            warmup_conversation_sets=1,
        )


def test_resident_runner_terminates_a_long_multiline_response_cycle(tmp_path) -> None:
    fake_runtime = tmp_path / "repeating_runtime.py"
    fake_runtime.write_text(
        """
import sys
import time

print("ready")
print("you> ", end="", flush=True)
sys.stdin.readline()
print("llm> reasoning")
cycle = "\\n".join(
    f"* There is a Corinth in region {index} with qualifier {index * 17}."
    for index in range(24)
)
print("\\n".join([cycle] * 20), flush=True)
time.sleep(0.2)
""".lstrip()
    )

    with pytest.raises(ConversationGateError, match="repeated suffix"):
        run_resident_conversation([sys.executable, "-u", str(fake_runtime)])
