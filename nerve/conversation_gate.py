from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

from nerve.text_quality import repeated_segment


WARMUP_PROMPT = "hi"
MEASURED_PROMPTS = (
    "Who are you?",
    "what is the capital of Greece?",
    'How many cities named "Corinth" are there?',
    "What is your knowledge cutoff date?",
    "I asked you earlier to tell me the capital of a country. Which country was that?",
)
CANONICAL_CONVERSATION_PROMPTS = (WARMUP_PROMPT, *MEASURED_PROMPTS)
CANONICAL_OUTPUT_TOKEN_ALLOWANCE = 65_536
_PROMPT_MARKER = b"you> "
_RESPONSE_PREFIX = "llm> "
_TURN_START = _PROMPT_MARKER.decode() + _RESPONSE_PREFIX
_TURN_ERROR_MARKER = b"\nturn_error: "
_NEW_CONVERSATION_COMMAND = "/new"
_SESSION_RESET_MARKER = b"session_reset: "
_STATS_MARKER = "\nstats:\n"
_STAT_LINE = re.compile(r"^  ([a-z][a-z0-9_]*)=(.+)$")
_RESIDENCY_POLICY = re.compile(r"^  policy=([^ ]+)", re.MULTILINE)
_RESIDENCY_COUNTER_LINE = re.compile(r"^  ([a-z_]+)\(([^)]+)\)=([^\n]+)$", re.MULTILINE)
_PHYSICAL_EXECUTION_SUMMARY_START = (
    "physical_execution=VulkanMountedPhysicalExecutionSummary {"
)
_PHYSICAL_EXECUTION_SUMMARY = re.compile(
    re.escape(_PHYSICAL_EXECUTION_SUMMARY_START) + r"(?P<body>[^{}\n]*)}"
)
_PHYSICAL_EXECUTION_FIELDS = (
    "tensor_parallel_island_count",
    "whole_expert_parallel_island_count",
    "intra_expert_tensor_parallel_island_count",
    "hybrid_island_count",
    "selected_resource_placement_count",
)

_CUMULATIVE_RESIDENCY_COUNTER_GROUPS = {
    "gpu_accesses",
    "residency_requests",
    "residency_eviction",
    "transfers",
}
_RESIDENCY_GAUGE_GROUPS = {"memory_tiers"}


class ConversationGateError(RuntimeError):
    pass


class ResidentConversationError(ConversationGateError):
    def __init__(self, message: str, transcript: str) -> None:
        super().__init__(message)
        self.transcript = transcript


@dataclass(frozen=True)
class PhysicalExecutionSummary:
    tensor_parallel_island_count: int
    whole_expert_parallel_island_count: int
    intra_expert_tensor_parallel_island_count: int
    hybrid_island_count: int
    selected_resource_placement_count: int

    @property
    def total_tensor_parallel_island_count(self) -> int:
        return (
            self.tensor_parallel_island_count
            + self.intra_expert_tensor_parallel_island_count
            + self.hybrid_island_count
        )


@dataclass(frozen=True)
class ConversationTurn:
    prompt: str
    response: str
    stats: dict[str, int | float | str]
    residency_policy: str | None = None
    residency_counters: dict[str, int | float] | None = None
    residency_gauges: dict[str, int | float] | None = None

    @property
    def decode_tokens_per_second(self) -> float:
        value = self.stats.get("decode_tokens_per_second")
        if not isinstance(value, (int, float)):
            raise ConversationGateError(
                f"turn {self.prompt!r} did not report decode_tokens_per_second"
            )
        return float(value)

    @property
    def prefill_tokens_per_second(self) -> float:
        value = self.stats.get("prefill_tokens_per_second")
        if not isinstance(value, (int, float)):
            raise ConversationGateError(
                f"turn {self.prompt!r} did not report prefill_tokens_per_second"
            )
        return float(value)


@dataclass(frozen=True)
class ConversationSetReport:
    warmup: ConversationTurn
    turns: list[ConversationTurn]
    mean_decode_tokens_per_second: float
    mean_prefill_tokens_per_second: float
    residency_policy: str | None
    residency_start: dict[str, int | float] | None
    residency_end: dict[str, int | float] | None
    residency_delta: dict[str, int | float] | None
    residency_gauges_start: dict[str, int | float] | None
    residency_gauges_end: dict[str, int | float] | None


@dataclass(frozen=True)
class ConversationSeedReport:
    seed: int
    command: list[str]
    transcript_sha256: str
    physical_execution: PhysicalExecutionSummary | None
    discarded_warmup_sets: list[ConversationSetReport]
    measured_set: ConversationSetReport


@dataclass(frozen=True)
class ConversationGateReport:
    ok: bool
    minimum_decode_tokens_per_second: float
    minimum_tensor_parallel_islands: int
    require_thinking: bool
    warmup_conversation_sets: int
    package: dict[str, Any]
    runs: list[ConversationSeedReport]


def _parse_scalar(raw: str) -> int | float | str:
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def _residency_metrics(
    report: str,
) -> tuple[
    str | None,
    dict[str, int | float] | None,
    dict[str, int | float] | None,
]:
    marker = "\nresource_residency:\n"
    if marker not in report:
        return None, None, None
    block = report.split(marker, 1)[1]
    policy_match = _RESIDENCY_POLICY.search(block)
    counters: dict[str, int | float] = {}
    gauges: dict[str, int | float] = {}
    for group, raw_labels, raw_values in _RESIDENCY_COUNTER_LINE.findall(block):
        if group not in (
            _CUMULATIVE_RESIDENCY_COUNTER_GROUPS | _RESIDENCY_GAUGE_GROUPS
        ):
            continue
        labels = raw_labels.split("/")
        values = raw_values.split("/")
        if len(labels) != len(values):
            raise ConversationGateError(
                f"resource residency group {group!r} has {len(labels)} labels "
                f"but {len(values)} values"
            )
        for label, value in zip(labels, values, strict=True):
            parsed = _parse_scalar(value)
            if not isinstance(parsed, (int, float)):
                raise ConversationGateError(
                    f"resource residency counter {group}.{label} is not numeric: "
                    f"{value!r}"
                )
            destination = (
                counters if group in _CUMULATIVE_RESIDENCY_COUNTER_GROUPS else gauges
            )
            destination[f"{group}.{label}"] = parsed
    return (
        policy_match.group(1) if policy_match is not None else None,
        counters or None,
        gauges or None,
    )


def parse_physical_execution_summary(
    transcript: str,
    *,
    minimum_tensor_parallel_islands: int = 0,
) -> PhysicalExecutionSummary | None:
    if minimum_tensor_parallel_islands < 0:
        raise ConversationGateError(
            "minimum tensor-parallel island count cannot be negative"
        )
    marker_count = transcript.count(_PHYSICAL_EXECUTION_SUMMARY_START)
    if marker_count == 0:
        if minimum_tensor_parallel_islands > 0:
            raise ConversationGateError(
                "runtime did not report mounted physical execution"
            )
        return None
    if marker_count != 1:
        raise ConversationGateError(
            "runtime reported more than one physical execution summary"
        )
    match = _PHYSICAL_EXECUTION_SUMMARY.search(transcript)
    if match is None:
        raise ConversationGateError("runtime physical execution summary is malformed")

    values: dict[str, int] = {}
    for entry in match.group("body").split(","):
        parts = entry.strip().split(":", 1)
        if len(parts) != 2:
            raise ConversationGateError(
                "runtime physical execution summary is malformed"
            )
        key, raw_value = (part.strip() for part in parts)
        if key not in _PHYSICAL_EXECUTION_FIELDS or key in values:
            raise ConversationGateError(
                "runtime physical execution summary has an unknown or duplicate field"
            )
        try:
            value = int(raw_value)
        except ValueError as error:
            raise ConversationGateError(
                f"runtime physical execution summary field {key!r} is not an integer"
            ) from error
        if value < 0:
            raise ConversationGateError(
                f"runtime physical execution summary field {key!r} is negative"
            )
        values[key] = value
    missing = set(_PHYSICAL_EXECUTION_FIELDS).difference(values)
    if missing:
        raise ConversationGateError(
            "runtime physical execution summary is missing field(s): "
            + ", ".join(sorted(missing))
        )

    summary = PhysicalExecutionSummary(**values)
    if summary.total_tensor_parallel_island_count < minimum_tensor_parallel_islands:
        raise ConversationGateError(
            "runtime mounted "
            f"{summary.total_tensor_parallel_island_count} tensor-parallel island(s); "
            f"required at least {minimum_tensor_parallel_islands}"
        )
    return summary


def parse_conversation_transcript(
    transcript: str,
    *,
    warmup_conversation_sets: int = 0,
) -> list[ConversationTurn]:
    if warmup_conversation_sets < 0:
        raise ConversationGateError("warmup conversation set count cannot be negative")
    completed_sections = []
    for section in transcript.split(_TURN_START)[1:]:
        if _STATS_MARKER not in section:
            continue
        response, report = section.rsplit(_STATS_MARKER, 1)
        stats: dict[str, int | float | str] = {}
        for line in report.splitlines():
            match = _STAT_LINE.match(line)
            if match is None:
                if stats:
                    break
                continue
            stats[match.group(1)] = _parse_scalar(match.group(2))
        policy, counters, gauges = _residency_metrics(report)
        completed_sections.append((response.rstrip(), stats, policy, counters, gauges))

    expected_prompts = CANONICAL_CONVERSATION_PROMPTS * (warmup_conversation_sets + 1)
    if len(completed_sections) != len(expected_prompts):
        raise ConversationGateError(
            "chat transcript contains "
            f"{len(completed_sections)} completed turn(s); expected {len(expected_prompts)}"
        )
    return [
        ConversationTurn(
            prompt=prompt,
            response=response,
            stats=stats,
            residency_policy=policy,
            residency_counters=counters,
            residency_gauges=gauges,
        )
        for prompt, (response, stats, policy, counters, gauges) in zip(
            expected_prompts, completed_sections, strict=True
        )
    ]


def _final_answer(response: str, require_thinking: bool) -> str:
    closing_count = response.count("</think>")
    opening_count = response.count("<think>")
    if require_thinking:
        if closing_count == 1:
            if opening_count > 1:
                raise ConversationGateError(
                    "thinking response contains more than one <think> boundary"
                )
            answer = response.rsplit("</think>", 1)[1].strip()
        elif (
            closing_count == 0
            and opening_count == 0
            and response.startswith(("thought\n", "analysis\n"))
        ):
            # Channel-based templates can decode the channel label while their
            # special delimiters are intentionally omitted by the tokenizer.
            # The model may not emit a second visible delimiter before its
            # answer, so retain the complete, validated channel stream.
            answer = response.split("\n", 1)[1].strip()
        else:
            raise ConversationGateError(
                "thinking response must contain one </think> boundary or begin "
                "with a decoded thought/analysis channel"
            )
    else:
        if closing_count > 1 or opening_count > 1:
            raise ConversationGateError(
                "response contains malformed thinking boundaries"
            )
        answer = response.rsplit("</think>", 1)[-1].strip()
    if not answer:
        raise ConversationGateError("response terminated without a final answer")
    return answer


def validate_conversation_turns(
    turns: Sequence[ConversationTurn],
    *,
    require_thinking: bool,
    minimum_decode_tokens_per_second: float,
) -> tuple[float, float]:
    if len(turns) != len(MEASURED_PROMPTS):
        raise ConversationGateError(
            f"expected {len(MEASURED_PROMPTS)} measured turns; found {len(turns)}"
        )

    answers = []
    for expected_prompt, turn in zip(MEASURED_PROMPTS, turns, strict=True):
        if turn.prompt != expected_prompt:
            raise ConversationGateError(
                f"expected prompt {expected_prompt!r}; found {turn.prompt!r}"
            )
        answer = _final_answer(turn.response, require_thinking)
        repeated = repeated_segment(turn.response)
        if repeated is not None:
            raise ConversationGateError(
                f"turn {turn.prompt!r} ends in a repeated segment: {repeated!r}"
            )
        answers.append(answer)

    if "athens" not in answers[1].casefold():
        raise ConversationGateError("capital-of-Greece turn did not answer Athens")
    if "corinth" not in answers[2].casefold():
        raise ConversationGateError(
            "Corinth turn did not answer the question about Corinth"
        )
    if "greece" not in answers[4].casefold():
        raise ConversationGateError(
            "conversation-recall turn did not identify Greece from prior history"
        )

    decode_rates = [turn.decode_tokens_per_second for turn in turns]
    prefill_rates = [turn.prefill_tokens_per_second for turn in turns]
    mean_decode = fmean(decode_rates)
    mean_prefill = fmean(prefill_rates)
    if mean_decode < minimum_decode_tokens_per_second:
        raise ConversationGateError(
            f"mean decode throughput {mean_decode:.3f} tok/s is below "
            f"the {minimum_decode_tokens_per_second:.3f} tok/s gate"
        )
    return mean_decode, mean_prefill


def _option_value(command: Sequence[str], option: str) -> str | None:
    positions = [index for index, value in enumerate(command) if value == option]
    if not positions:
        return None
    index = positions[-1]
    if index + 1 >= len(command):
        raise ConversationGateError(f"{option} is missing its value")
    return command[index + 1]


def _replace_option(command: Sequence[str], option: str, value: str) -> list[str]:
    replaced: list[str] = []
    cursor = 0
    while cursor < len(command):
        if command[cursor] == option:
            if cursor + 1 >= len(command):
                raise ConversationGateError(f"{option} is missing its value")
            cursor += 2
            continue
        replaced.append(command[cursor])
        cursor += 1
    replaced.extend((option, value))
    return replaced


def canonical_runtime_command(command: Sequence[str], seed: int) -> list[str]:
    if not command:
        raise ConversationGateError("runtime command must not be empty")
    if "--chat" not in command:
        raise ConversationGateError(
            "conversation gate requires the normal --chat runtime mode"
        )
    if "--prompt" in command:
        raise ConversationGateError(
            "conversation gate owns the canonical warmup and measured prompts"
        )
    if "--json" in command or "--generated-only" in command:
        raise ConversationGateError(
            "conversation gate requires the normal default chat output and statistics"
        )
    raw_limit = _option_value(command, "--max-new-tokens")
    if raw_limit is not None:
        try:
            limit = int(raw_limit)
        except ValueError as error:
            raise ConversationGateError(
                f"invalid --max-new-tokens value {raw_limit!r}"
            ) from error
        if limit != CANONICAL_OUTPUT_TOKEN_ALLOWANCE:
            raise ConversationGateError(
                "conversation gate requires --max-new-tokens "
                f"{CANONICAL_OUTPUT_TOKEN_ALLOWANCE}; found {limit}"
            )
    else:
        command = (*command, "--max-new-tokens", str(CANONICAL_OUTPUT_TOKEN_ALLOWANCE))
    return _replace_option(command, "--seed", str(seed))


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_resident_conversation(
    command: Sequence[str],
    *,
    warmup_conversation_sets: int = 0,
) -> tuple[str, int]:
    if warmup_conversation_sets < 0:
        raise ConversationGateError("warmup conversation set count cannot be negative")
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    if process.stdin is None or process.stdout is None:
        _terminate(process)
        raise ConversationGateError("could not open runtime process pipes")

    prompts: list[str] = []
    for set_index in range(warmup_conversation_sets + 1):
        prompts.extend(CANONICAL_CONVERSATION_PROMPTS)
        if set_index < warmup_conversation_sets:
            prompts.append(_NEW_CONVERSATION_COMMAND)
    prompts.append("/exit")
    transcript = bytearray()
    search_from = 0
    accepted_marker_end = 0
    sent = 0
    checked_response_bytes = 0
    live_error: str | None = None
    try:
        while True:
            chunk = os.read(process.stdout.fileno(), 65_536)
            if not chunk:
                break
            transcript.extend(chunk)
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()

            turn_error = transcript.find(_TURN_ERROR_MARKER, accepted_marker_end)
            if turn_error >= 0:
                detail_start = turn_error + len(_TURN_ERROR_MARKER)
                detail_end = transcript.find(b"\n", detail_start)
                if detail_end >= 0:
                    detail = transcript[detail_start:detail_end].decode(
                        errors="replace"
                    )
                    live_error = f"runtime rejected a recoverable chat turn: {detail}"
                    break

            while True:
                marker = transcript.find(_PROMPT_MARKER, search_from)
                if marker < 0:
                    search_from = max(0, len(transcript) - len(_PROMPT_MARKER) + 1)
                    break
                search_from = marker + len(_PROMPT_MARKER)
                if sent > 0:
                    previous_output = transcript[accepted_marker_end:marker]
                    if prompts[sent - 1] == _NEW_CONVERSATION_COMMAND:
                        if _SESSION_RESET_MARKER not in previous_output:
                            live_error = (
                                "runtime did not acknowledge the required "
                                "new-conversation reset"
                            )
                            break
                    elif _STATS_MARKER.encode() not in previous_output:
                        continue
                if sent >= len(prompts):
                    live_error = (
                        "runtime requested more chat turns than the gate supplied"
                    )
                    break
                process.stdin.write((prompts[sent] + "\n").encode())
                process.stdin.flush()
                accepted_marker_end = marker + len(_PROMPT_MARKER)
                sent += 1
            if live_error is not None:
                break

            latest_prompt = transcript.rfind(_PROMPT_MARKER)
            response_start = transcript.find(
                _RESPONSE_PREFIX.encode(),
                latest_prompt + len(_PROMPT_MARKER),
            )
            stats_start = transcript.find(
                _STATS_MARKER.encode(),
                response_start + len(_RESPONSE_PREFIX) if response_start >= 0 else 0,
            )
            if (
                response_start >= 0
                and stats_start < 0
                and len(transcript) - checked_response_bytes >= 16_384
            ):
                response = transcript[response_start + len(_RESPONSE_PREFIX) :].decode(
                    errors="replace"
                )
                repeated = repeated_segment(response)
                checked_response_bytes = len(transcript)
                if repeated is not None:
                    live_error = (
                        "runtime response entered a repeated segment before termination: "
                        f"{repeated!r}"
                    )
                    break
        if live_error is not None:
            _terminate(process)
        return_code = process.wait()
    except BaseException:
        _terminate(process)
        raise
    finally:
        process.stdin.close()
        process.stdout.close()

    decoded_transcript = transcript.decode(errors="replace")
    if live_error is not None:
        raise ResidentConversationError(live_error, decoded_transcript)
    if return_code != 0:
        output_tail = bytes(transcript[-4_096:]).decode(errors="replace").strip()
        detail = f"; output tail:\n{output_tail}" if output_tail else ""
        raise ResidentConversationError(
            f"runtime exited with status {return_code}{detail}",
            decoded_transcript,
        )
    if sent != len(prompts):
        raise ResidentConversationError(
            f"runtime accepted {sent} scripted input(s); expected {len(prompts)}",
            decoded_transcript,
        )
    return decoded_transcript, return_code


def _package_metadata(command: Sequence[str]) -> dict[str, Any]:
    raw_path = _option_value(command, "--package")
    if raw_path is None:
        raise ConversationGateError("runtime command is missing --package")
    path = Path(raw_path).expanduser().resolve()
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ConversationGateError(
            f"could not read compiled package manifest {path}: {error}"
        ) from error
    shader_paths = {
        stage["shader_path"]
        for component in manifest.get("component_executions", [])
        for kernel in component.get("kernels", [])
        for implementation in (
            [{"stages": [{"shader_path": kernel.get("shader_path")}]}]
            + kernel.get("batch_implementations", [])
        )
        for stage in implementation.get("stages", [])
        if stage.get("shader_path")
    }
    return {
        "manifest": str(path),
        "package_id": manifest.get("package_id"),
        "package_schema": manifest.get("schema"),
        "compiler_target": manifest.get("compiler_target"),
        "compiled_shader_variant_count": len(shader_paths),
    }


def run_conversation_gate(
    command: Sequence[str],
    *,
    seeds: Sequence[int],
    minimum_decode_tokens_per_second: float,
    require_thinking: bool,
    minimum_tensor_parallel_islands: int = 0,
    warmup_conversation_sets: int = 0,
    transcript_dir: Path | None = None,
) -> ConversationGateReport:
    if not seeds:
        raise ConversationGateError("at least one fixed sampler seed is required")
    if len(seeds) != 1:
        raise ConversationGateError(
            "run exactly one seed per invocation so GPU residency can be verified "
            "between model loads"
        )
    if warmup_conversation_sets < 0:
        raise ConversationGateError("warmup conversation set count cannot be negative")
    if minimum_tensor_parallel_islands < 0:
        raise ConversationGateError(
            "minimum tensor-parallel island count cannot be negative"
        )
    package = _package_metadata(command)
    runs = []
    for seed in seeds:
        seeded_command = canonical_runtime_command(command, seed)
        try:
            transcript, _ = run_resident_conversation(
                seeded_command,
                warmup_conversation_sets=warmup_conversation_sets,
            )
        except ResidentConversationError as error:
            if transcript_dir is not None:
                transcript_dir.mkdir(parents=True, exist_ok=True)
                (transcript_dir / f"conversation-seed-{seed}-failed.log").write_text(
                    error.transcript
                )
            raise
        if transcript_dir is not None:
            transcript_dir.mkdir(parents=True, exist_ok=True)
            (transcript_dir / f"conversation-seed-{seed}.log").write_text(transcript)
        parsed = parse_conversation_transcript(
            transcript,
            warmup_conversation_sets=warmup_conversation_sets,
        )
        physical_execution = parse_physical_execution_summary(
            transcript,
            minimum_tensor_parallel_islands=minimum_tensor_parallel_islands,
        )
        conversation_sets = [
            parsed[index : index + len(CANONICAL_CONVERSATION_PROMPTS)]
            for index in range(0, len(parsed), len(CANONICAL_CONVERSATION_PROMPTS))
        ]
        reports: list[ConversationSetReport] = []
        prior_residency: dict[str, int | float] | None = None
        prior_residency_gauges: dict[str, int | float] | None = None
        for set_index, conversation_set in enumerate(conversation_sets):
            warmup, turns = conversation_set[0], conversation_set[1:]
            _final_answer(warmup.response, require_thinking)
            if repeated_segment(warmup.response) is not None:
                raise ConversationGateError(
                    f"seed {seed} conversation set {set_index + 1} "
                    "warmup ended in repetition"
                )
            set_minimum = (
                minimum_decode_tokens_per_second
                if set_index == warmup_conversation_sets
                else 0.0
            )
            mean_decode, mean_prefill = validate_conversation_turns(
                turns,
                require_thinking=require_thinking,
                minimum_decode_tokens_per_second=set_minimum,
            )
            residency_end = conversation_set[-1].residency_counters
            residency_gauges_end = conversation_set[-1].residency_gauges
            residency_delta = _residency_delta(prior_residency, residency_end)
            residency_policies = {
                turn.residency_policy
                for turn in conversation_set
                if turn.residency_policy is not None
            }
            if len(residency_policies) > 1:
                raise ConversationGateError(
                    "resource residency policy changed within one conversation set"
                )
            reports.append(
                ConversationSetReport(
                    warmup=warmup,
                    turns=list(turns),
                    mean_decode_tokens_per_second=mean_decode,
                    mean_prefill_tokens_per_second=mean_prefill,
                    residency_policy=(
                        next(iter(residency_policies)) if residency_policies else None
                    ),
                    residency_start=prior_residency,
                    residency_end=residency_end,
                    residency_delta=residency_delta,
                    residency_gauges_start=prior_residency_gauges,
                    residency_gauges_end=residency_gauges_end,
                )
            )
            prior_residency = residency_end
            prior_residency_gauges = residency_gauges_end
        session_policies = {
            report.residency_policy
            for report in reports
            if report.residency_policy is not None
        }
        if len(session_policies) > 1:
            raise ConversationGateError(
                "resource residency policy changed within one resident session"
            )
        runs.append(
            ConversationSeedReport(
                seed=seed,
                command=seeded_command,
                transcript_sha256=hashlib.sha256(transcript.encode()).hexdigest(),
                physical_execution=physical_execution,
                discarded_warmup_sets=reports[:-1],
                measured_set=reports[-1],
            )
        )
    return ConversationGateReport(
        ok=True,
        minimum_decode_tokens_per_second=minimum_decode_tokens_per_second,
        minimum_tensor_parallel_islands=minimum_tensor_parallel_islands,
        require_thinking=require_thinking,
        warmup_conversation_sets=warmup_conversation_sets,
        package=package,
        runs=runs,
    )


def _parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from error
    if not seeds or any(seed < 0 or seed > 0xFFFF_FFFF for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must contain one or more U32 values")
    return seeds


def _residency_delta(
    start: dict[str, int | float] | None,
    end: dict[str, int | float] | None,
) -> dict[str, int | float] | None:
    if end is None:
        return None
    if start is None:
        return dict(end)
    if start.keys() != end.keys():
        raise ConversationGateError(
            "resource residency counter schema changed within one resident session"
        )
    delta: dict[str, int | float] = {}
    for key, end_value in end.items():
        start_value = start[key]
        if end_value < start_value:
            raise ConversationGateError(
                f"cumulative resource residency counter {key!r} decreased from "
                f"{start_value} to {end_value}"
            )
        delta[key] = end_value - start_value
    return delta


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the canonical warmup plus five-turn correctness/performance gate "
            "through nerve-runtime's normal resident chat mode."
        )
    )
    parser.add_argument("--seeds", type=_parse_seeds, default=(0,))
    parser.add_argument("--minimum-decode-tps", type=float, default=0.0)
    parser.add_argument("--minimum-tensor-parallel-islands", type=int, default=0)
    parser.add_argument("--require-thinking", action="store_true")
    parser.add_argument(
        "--warmup-conversation-sets",
        type=int,
        default=0,
        help=(
            "run and discard this many complete canonical conversations in the "
            "same resident process before measuring the final conversation"
        ),
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--transcript-dir", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        report = run_conversation_gate(
            command,
            seeds=args.seeds,
            minimum_decode_tokens_per_second=args.minimum_decode_tps,
            minimum_tensor_parallel_islands=args.minimum_tensor_parallel_islands,
            require_thinking=args.require_thinking,
            warmup_conversation_sets=args.warmup_conversation_sets,
            transcript_dir=args.transcript_dir,
        )
    except ConversationGateError as error:
        print(f"conversation gate failed: {error}", file=sys.stderr)
        return 1

    encoded = json.dumps(asdict(report), indent=2, sort_keys=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
