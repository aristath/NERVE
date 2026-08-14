from __future__ import annotations

import json
import sys

import pytest

from nerve.conversation_gate import (
    CANONICAL_CONVERSATION_PROMPTS,
    CANONICAL_OUTPUT_TOKEN_ALLOWANCE,
    MEASURED_PROMPTS,
    WARMUP_PROMPT,
    _validate_shutdown_device_restoration,
    ConversationGateError,
    ConversationTurn,
    canonical_runtime_command,
    parse_conversation_transcript,
    parse_device_restoration_report,
    parse_physical_execution_summary,
    parse_shutdown_report,
    run_conversation_gate,
    run_resident_conversation,
    validate_conversation_turns,
)
from nerve.device_reservation_gate import (
    EXTERNAL_DEVICE_RESERVATION_SNAPSHOT_SCHEMA,
    DrmDeviceReservation,
    DrmProcessReservation,
    ExternalDeviceReservationSnapshot,
)
from nerve.text_quality import repeated_segment


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


def _tp_execution_counters(
    *,
    decode_tp: int = 1,
    prefill_tp: int = 1,
    decode_whole_expert: int = 0,
    prefill_whole_expert: int = 0,
) -> dict[str, int]:
    decode_islands = decode_tp + decode_whole_expert
    prefill_islands = prefill_tp + prefill_whole_expert
    return {
        "distributed_decode_island_submissions": decode_islands,
        "distributed_decode_shard_submissions": decode_islands * 2,
        "distributed_decode_tensor_parallel_island_submissions": decode_tp,
        "distributed_decode_whole_expert_parallel_island_submissions": (
            decode_whole_expert
        ),
        "distributed_decode_intra_expert_tensor_parallel_island_submissions": 0,
        "distributed_decode_hybrid_island_submissions": 0,
        "distributed_prefill_island_submissions": prefill_islands,
        "distributed_prefill_shard_submissions": prefill_islands * 2,
        "distributed_prefill_tensor_parallel_island_submissions": prefill_tp,
        "distributed_prefill_whole_expert_parallel_island_submissions": (
            prefill_whole_expert
        ),
        "distributed_prefill_intra_expert_tensor_parallel_island_submissions": 0,
        "distributed_prefill_hybrid_island_submissions": 0,
    }


def _valid_shutdown_transcript() -> str:
    return (
        "ready\nyou> shutdown:\n"
        "  complete=true streams=1 packages=1 scheduler_in_flight=0\n"
        "  physical_devices_acknowledged=2/2 released_units=3 "
        "released_payload_bytes=96 cancelled_loads=1\n"
        "  package=fixture scope=target physical_devices_acknowledged=2/2\n"
        "  store=store0 physical_device=gpu0 acknowledged=true "
        "remaining_units=0 remaining_payload_bytes=0 error=None\n"
        "  store=store1 physical_device=gpu1 acknowledged=true "
        "remaining_units=0 remaining_payload_bytes=0 error=None\n"
        + _valid_device_restoration_transcript(("gpu0", "gpu1"))
    )


def _test_pci_address(physical_device_id: str) -> str:
    suffix = int(physical_device_id.removeprefix("gpu"))
    return f"0000:{suffix + 1:02x}:00.0"


def _device_restoration_snapshot(physical_device_id: str) -> dict[str, object]:
    return {
        "physical_device_id": physical_device_id,
        "device_name": "test device",
        "pci_address": _test_pci_address(physical_device_id),
        "api_version": 1,
        "driver_version": 2,
        "heap_index": 1,
        "physical_heap_bytes": 2_000,
        "memory_budget_supported": True,
        "budget_bytes": 1_000,
        "usage_bytes": 100,
        "available_bytes": 900,
    }


def _device_restoration_payload(
    physical_device_ids: tuple[str, ...] = ("gpu0",),
) -> dict[str, object]:
    devices = []
    for physical_device_id in physical_device_ids:
        snapshot = _device_restoration_snapshot(physical_device_id)
        devices.append(
            {
                "physical_device_id": physical_device_id,
                "restored": True,
                "usage_counter_tolerance_bytes": 16,
                "before": snapshot,
                "after": json.loads(json.dumps(snapshot)),
                "errors": [],
            }
        )
    return {
        "schema": "nerve.runtime.physical_device_memory_restoration.v2",
        "complete": True,
        "physical_device_count": len(devices),
        "restored_device_count": len(devices),
        "devices": devices,
        "errors": [],
    }


def _valid_device_restoration_transcript(
    physical_device_ids: tuple[str, ...] = ("gpu0",),
) -> str:
    return (
        "device_restoration:\n  "
        + json.dumps(
            _device_restoration_payload(physical_device_ids),
            separators=(",", ":"),
        )
        + "\n"
    )


class _StaticDeviceReservationProbe:
    def __init__(
        self,
        *snapshots: ExternalDeviceReservationSnapshot,
    ) -> None:
        assert snapshots
        self.snapshots = snapshots
        self.capture_count = 0

    def capture(self) -> ExternalDeviceReservationSnapshot:
        snapshot = self.snapshots[min(self.capture_count, len(self.snapshots) - 1)]
        self.capture_count += 1
        return snapshot


def _external_reservation_snapshot(
    physical_device_ids: tuple[str, ...] = ("gpu0",),
    *,
    used_delta: int = 0,
    processes: tuple[DrmProcessReservation, ...] = (),
) -> ExternalDeviceReservationSnapshot:
    return ExternalDeviceReservationSnapshot(
        schema=EXTERNAL_DEVICE_RESERVATION_SNAPSHOT_SCHEMA,
        devices=tuple(
            DrmDeviceReservation(
                pci_address=_test_pci_address(physical_device_id),
                drm_card=f"card{index}",
                vram_total_bytes=32 * 1024**3,
                vram_used_bytes=100 * 1024**2 + used_delta,
                busy_percent=0,
                resident_processes=processes,
            )
            for index, physical_device_id in enumerate(physical_device_ids)
        ),
    )


def _reservation_probe(
    physical_device_ids: tuple[str, ...] = ("gpu0",),
) -> _StaticDeviceReservationProbe:
    snapshot = _external_reservation_snapshot(physical_device_ids)
    return _StaticDeviceReservationProbe(snapshot, snapshot)


def _turn_with_execution(
    prompt: str,
    answer: str,
    execution_counters: dict[str, int],
    decode: float = 25.0,
) -> ConversationTurn:
    return ConversationTurn(
        prompt=prompt,
        response=_response(answer),
        stats={
            "decode_tokens_per_second": decode,
            "prefill_tokens_per_second": 100.0,
        },
        execution_counters=execution_counters,
    )


def _valid_turns() -> list[ConversationTurn]:
    return [
        _turn(MEASURED_PROMPTS[0], "I am a language model."),
        _turn(MEASURED_PROMPTS[1], "The capital is Athens."),
        _turn(MEASURED_PROMPTS[2], "Corinth refers to cities in Greece and elsewhere."),
        _turn(MEASURED_PROMPTS[3], "My knowledge cutoff is not available."),
        _turn(MEASURED_PROMPTS[4], "You asked about Greece."),
    ]


def test_physical_execution_summary_proves_mounted_tensor_parallel_work() -> None:
    summary = parse_physical_execution_summary(
        "nerve chat ready: physical_execution=VulkanMountedPhysicalExecutionSummary { "
        "tensor_parallel_island_count: 2, whole_expert_parallel_island_count: 3, "
        "intra_expert_tensor_parallel_island_count: 5, hybrid_island_count: 7, "
        "selected_resource_placement_count: 11 }, setup_ms=12.000\n",
        minimum_tensor_parallel_islands=14,
    )

    assert summary is not None
    assert summary.tensor_parallel_island_count == 2
    assert summary.intra_expert_tensor_parallel_island_count == 5
    assert summary.hybrid_island_count == 7
    assert summary.total_tensor_parallel_island_count == 14


def test_physical_execution_summary_rejects_missing_required_tp_proof() -> None:
    with pytest.raises(
        ConversationGateError, match="did not report mounted physical execution"
    ):
        parse_physical_execution_summary(
            "nerve chat ready: setup_ms=12.000\n",
            minimum_tensor_parallel_islands=1,
        )


def test_physical_execution_summary_rejects_whole_expert_parallel_as_tp() -> None:
    with pytest.raises(ConversationGateError, match="0 tensor-parallel island"):
        parse_physical_execution_summary(
            "nerve chat ready: physical_execution=VulkanMountedPhysicalExecutionSummary { "
            "tensor_parallel_island_count: 0, whole_expert_parallel_island_count: 4, "
            "intra_expert_tensor_parallel_island_count: 0, hybrid_island_count: 0, "
            "selected_resource_placement_count: 1 }, setup_ms=12.000\n",
            minimum_tensor_parallel_islands=1,
        )


@pytest.mark.parametrize(
    "transcript",
    (
        "physical_execution=VulkanMountedPhysicalExecutionSummary { tensor_parallel_island_count: nope }",
        (
            "physical_execution=VulkanMountedPhysicalExecutionSummary { "
            "tensor_parallel_island_count: 1, tensor_parallel_island_count: 2, "
            "whole_expert_parallel_island_count: 0, "
            "intra_expert_tensor_parallel_island_count: 0, hybrid_island_count: 0, "
            "selected_resource_placement_count: 0 }"
        ),
    ),
)
def test_physical_execution_summary_rejects_malformed_or_ambiguous_proof(
    transcript: str,
) -> None:
    with pytest.raises(ConversationGateError, match="physical execution summary"):
        parse_physical_execution_summary(
            transcript,
            minimum_tensor_parallel_islands=1,
        )


def test_shutdown_report_proves_complete_resource_release() -> None:
    shutdown = parse_shutdown_report(_valid_shutdown_transcript())

    assert shutdown.complete
    assert shutdown.scheduler_in_flight_activation_count == 0
    assert shutdown.acknowledged_device_count == 2
    assert shutdown.physical_device_count == 2
    assert shutdown.released_unit_count == 3
    assert shutdown.released_payload_bytes == 96
    assert shutdown.cancelled_load_count == 1
    assert [device.physical_device_id for device in shutdown.packages[0].devices] == [
        "gpu0",
        "gpu1",
    ]


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        ("complete=true", "complete=false", "incomplete shutdown"),
        ("scheduler_in_flight=0", "scheduler_in_flight=1", "scheduler activations"),
        (
            "physical_devices_acknowledged=2/2 released_units",
            "physical_devices_acknowledged=1/2 released_units",
            "not acknowledged by every physical",
        ),
        (
            "package=fixture scope=target physical_devices_acknowledged=2/2",
            "package=fixture scope=target physical_devices_acknowledged=1/2",
            "package device totals",
        ),
        ("remaining_units=0", "remaining_units=1", "store incomplete"),
        (
            "remaining_payload_bytes=0",
            "remaining_payload_bytes=1",
            "store incomplete",
        ),
        ("error=None", 'error=Some("teardown failed")', "store incomplete"),
        ("store=store1", "store=store0", "repeats a physical resource store"),
        ("packages=1", "packages=2", "exactly one stream and one package"),
    ),
)
def test_shutdown_report_rejects_incomplete_or_incoherent_teardown(
    old: str,
    new: str,
    message: str,
) -> None:
    transcript = _valid_shutdown_transcript().replace(old, new, 1)

    with pytest.raises(ConversationGateError, match=message):
        parse_shutdown_report(transcript)


def test_shutdown_report_rejects_missing_or_duplicate_reports() -> None:
    with pytest.raises(
        ConversationGateError, match="did not report structured shutdown"
    ):
        parse_shutdown_report("ready\nyou> ")

    duplicated = _valid_shutdown_transcript() + _valid_shutdown_transcript()
    with pytest.raises(ConversationGateError, match="more than once"):
        parse_shutdown_report(duplicated)


def test_device_restoration_report_proves_all_selected_devices_restored() -> None:
    report = parse_device_restoration_report(
        _valid_device_restoration_transcript(("gpu0", "gpu1"))
    )

    assert report.complete
    assert report.physical_device_count == 2
    assert report.restored_device_count == 2
    assert [device.physical_device_id for device in report.devices] == [
        "gpu0",
        "gpu1",
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda payload: payload.update(complete=False),
            "incomplete device restoration",
        ),
        (
            lambda payload: payload.update(restored_device_count=0),
            "did not restore every selected",
        ),
        (
            lambda payload: payload["devices"][0].update(restored=False),
            "left 'gpu0' incomplete",
        ),
        (
            lambda payload: payload.update(errors=["failure"]),
            "reported global errors",
        ),
        (
            lambda payload: payload["devices"][0]["after"].update(
                usage_bytes=117,
                available_bytes=883,
            ),
            "did not restore usage_bytes",
        ),
        (
            lambda payload: payload["devices"][0]["after"].update(driver_version=3),
            "changed physical identity",
        ),
        (
            lambda payload: payload["devices"][0]["after"].update(heap_index=2),
            "changed physical identity",
        ),
    ),
)
def test_device_restoration_report_rejects_incomplete_or_false_proof(
    mutate,
    message: str,
) -> None:
    payload = _device_restoration_payload()
    mutate(payload)
    transcript = "device_restoration:\n  " + json.dumps(payload) + "\n"

    with pytest.raises(ConversationGateError, match=message):
        parse_device_restoration_report(transcript)


def test_device_restoration_report_accepts_non_invariant_budget_drift() -> None:
    payload = _device_restoration_payload()
    payload["devices"][0]["after"].update(
        budget_bytes=600,
        available_bytes=500,
    )

    report = parse_device_restoration_report(
        "device_restoration:\n  " + json.dumps(payload) + "\n"
    )

    assert report.complete
    assert report.devices[0].before["budget_bytes"] == 1_000
    assert report.devices[0].after["budget_bytes"] == 600


def test_device_restoration_report_rejects_missing_duplicate_and_unknown_schema() -> (
    None
):
    with pytest.raises(ConversationGateError, match="did not report"):
        parse_device_restoration_report("ready\n")

    duplicate = _valid_device_restoration_transcript() * 2
    with pytest.raises(ConversationGateError, match="more than once"):
        parse_device_restoration_report(duplicate)

    payload = _device_restoration_payload()
    payload["unknown"] = True
    with pytest.raises(ConversationGateError, match="invalid schema"):
        parse_device_restoration_report(
            "device_restoration:\n  " + json.dumps(payload) + "\n"
        )


def test_device_restoration_report_rejects_duplicate_physical_identity() -> None:
    payload = _device_restoration_payload(("gpu0", "gpu1"))
    payload["devices"][1]["physical_device_id"] = "gpu0"
    payload["devices"][1]["before"]["physical_device_id"] = "gpu0"
    payload["devices"][1]["after"]["physical_device_id"] = "gpu0"

    with pytest.raises(ConversationGateError, match="repeats a physical device"):
        parse_device_restoration_report(
            "device_restoration:\n  " + json.dumps(payload) + "\n"
        )


def test_device_restoration_must_cover_every_shutdown_resource_device() -> None:
    shutdown = parse_shutdown_report(_valid_shutdown_transcript())
    restoration = parse_device_restoration_report(
        _valid_device_restoration_transcript(("gpu0",))
    )

    with pytest.raises(ConversationGateError, match="omits a physical resource"):
        _validate_shutdown_device_restoration(shutdown, restoration)


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
    assert turns[-1].execution_counters == {
        "resident_sequence_queue_submits": 1,
    }


def test_tensor_parallel_validation_requires_decode_and_prefill_on_every_turn() -> None:
    answers = (
        "I am a language model.",
        "The capital is Athens.",
        "Corinth refers to cities in Greece and elsewhere.",
        "My knowledge cutoff is not available.",
        "You asked about Greece.",
    )
    turns = [
        _turn_with_execution(prompt, answer, _tp_execution_counters())
        for prompt, answer in zip(MEASURED_PROMPTS, answers, strict=True)
    ]

    validate_conversation_turns(
        turns,
        require_thinking=True,
        minimum_decode_tokens_per_second=20.0,
        require_tensor_parallel_execution=True,
    )

    turns[-1] = _turn_with_execution(
        MEASURED_PROMPTS[-1],
        answers[-1],
        _tp_execution_counters(prefill_tp=0),
    )
    with pytest.raises(
        ConversationGateError,
        match="did not submit a tensor-parallel prefill island",
    ):
        validate_conversation_turns(
            turns,
            require_thinking=True,
            minimum_decode_tokens_per_second=20.0,
            require_tensor_parallel_execution=True,
        )


def test_tensor_parallel_validation_does_not_count_whole_expert_parallelism() -> None:
    turns = _valid_turns()
    turns[0] = _turn_with_execution(
        MEASURED_PROMPTS[0],
        "I am a language model.",
        _tp_execution_counters(
            decode_tp=0,
            prefill_tp=0,
            decode_whole_expert=2,
            prefill_whole_expert=2,
        ),
    )

    with pytest.raises(
        ConversationGateError,
        match="did not submit a tensor-parallel decode island",
    ):
        validate_conversation_turns(
            turns,
            require_thinking=True,
            minimum_decode_tokens_per_second=20.0,
            require_tensor_parallel_execution=True,
        )


def test_gate_rejects_mounted_but_unused_tensor_parallelism(
    monkeypatch, tmp_path
) -> None:
    package = tmp_path / "package.json"
    package.write_text("{}")
    sections = [
        "nerve chat ready: physical_execution=VulkanMountedPhysicalExecutionSummary { "
        "tensor_parallel_island_count: 1, whole_expert_parallel_island_count: 0, "
        "intra_expert_tensor_parallel_island_count: 0, hybrid_island_count: 0, "
        "selected_resource_placement_count: 0 }, setup_ms=12.000\n"
    ]
    for answer in (
        "Hello.",
        "I am a language model.",
        "The capital of Greece is Athens.",
        "There are several cities named Corinth.",
        "My knowledge cutoff is unavailable.",
        "The country was Greece.",
    ):
        sections.append(
            "you> llm> "
            f"{_response(answer)}\n"
            "stats:\n"
            "  prefill_tokens_per_second=100.000\n"
            "  decode_tokens_per_second=25.000\n"
            "execution:\n"
            "  resident_sequence_queue_submits=1\n"
        )
    sections.append("you> ")
    transcript = "".join(sections)
    monkeypatch.setattr(
        "nerve.conversation_gate.run_resident_conversation",
        lambda command, warmup_conversation_sets=0, **_kwargs: (transcript, 0),
    )

    with pytest.raises(
        ConversationGateError,
        match="did not report execution counter 'distributed_decode_island_submissions'",
    ):
        run_conversation_gate(
            [
                sys.executable,
                "--package",
                str(package),
                "--chat",
            ],
            seeds=(0,),
            minimum_decode_tokens_per_second=20.0,
            minimum_tensor_parallel_islands=1,
            require_thinking=True,
            device_reservation_probe=_reservation_probe(),
        )


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

    turns = parse_conversation_transcript("".join(sections), warmup_conversation_sets=1)

    assert [turn.prompt for turn in turns] == list(CANONICAL_CONVERSATION_PROMPTS * 2)


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
            f"  payload_bytes(initial/current/high_water/maximum)=0/{index * 10}/{index * 10}/100 units(initial/current/high_water/addressable)=0/{index}/{index}/10\n"
            f"  gpu_accesses(selections/resident_hits/misses)={index * 2}/{index}/{index}\n"
            f"  residency_requests(directory_hits/load_required/deduplicated/succeeded/failed/cancelled)={index}/{index}/0/{index}/0/0\n"
            "  residency_eviction(cycles/units/payload_bytes/device_bytes/reloads)=0/0/0/0/0\n"
            f"  memory_tiers(device_payload/host_visible_payload/device_capacity/host_visible_capacity)={index * 8}/{index * 2}/80/20\n"
            f"  transfers(reads/source_bytes/resident_bytes/uploaded_bytes/read_ms/derivation_ms/upload_ms/blocking_ms)={index}/{index * 10}/{index * 20}/{index * 20}/{index * 0.5}/{index * 0.125}/{index * 0.25}/{index * 0.75}\n"
            "determinism:\n"
            f"  generated_tokens=nerve.runtime.token_ids_sha256.v1:{index:064x}\n"
            f"  selection_counters=nerve.runtime.selection_counters_sha256.v1:{index + 100:064x}\n"
            f"  resident_state=nerve.optimizer.artifact_sha256.v1:{index + 200:064x}\n"
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
        "payload_bytes.initial": 0,
        "payload_bytes.current": 60,
        "payload_bytes.high_water": 60,
        "payload_bytes.maximum": 100,
        "units.initial": 0,
        "units.current": 6,
        "units.high_water": 6,
        "units.addressable": 10,
        "memory_tiers.device_payload": 48,
        "memory_tiers.host_visible_payload": 12,
        "memory_tiers.device_capacity": 80,
        "memory_tiers.host_visible_capacity": 20,
    }
    assert turns[-1].execution_counters == {}


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
def test_repeated_segment_catches_text_token_and_unicode_loops(text: str) -> None:
    assert repeated_segment(text) is not None


def test_repeated_segment_catches_long_multiline_cycles() -> None:
    cycle = "\n".join(
        f"* There is a Corinth in region {index} with qualifier {index * 17}."
        for index in range(24)
    )

    repeated = repeated_segment("Reasoning begins.\n" + "\n".join([cycle] * 5))

    assert repeated is not None
    assert "Corinth" in repeated


def test_repeated_segment_catches_a_long_cycle_before_a_partial_next_cycle() -> None:
    cycle = " ".join(
        f"Corinth candidate {index} has distinct qualifier {index * 17}."
        for index in range(32)
    )
    response = "Reasoning begins. " + cycle * 4 + cycle[:311]

    repeated = repeated_segment(response)

    assert repeated is not None
    assert "Corinth" in repeated
    assert len(repeated) <= 256


def test_repeated_segment_allows_long_nonrepeating_reasoning() -> None:
    text = " ".join(f"distinct-step-{index}" for index in range(1_000))
    assert repeated_segment(text) is None


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


@pytest.mark.parametrize(
    ("statistic", "value"),
    (
        ("decode_tokens_per_second", float("nan")),
        ("decode_tokens_per_second", -1.0),
        ("prefill_tokens_per_second", float("inf")),
    ),
)
def test_gate_rejects_invalid_throughput_telemetry(
    statistic: str,
    value: float,
) -> None:
    turns = _valid_turns()
    stats = dict(turns[0].stats)
    stats[statistic] = value
    turns[0] = ConversationTurn(
        prompt=turns[0].prompt,
        response=turns[0].response,
        stats=stats,
    )

    with pytest.raises(ConversationGateError, match=f"invalid {statistic}"):
        validate_conversation_turns(
            turns,
            require_thinking=True,
            minimum_decode_tokens_per_second=20.0,
        )


@pytest.mark.parametrize("minimum", (float("nan"), float("inf"), -1.0))
def test_gate_rejects_invalid_throughput_floor(minimum: float) -> None:
    with pytest.raises(ConversationGateError, match="finite non-negative"):
        validate_conversation_turns(
            _valid_turns(),
            require_thinking=True,
            minimum_decode_tokens_per_second=minimum,
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


def test_resident_runner_stops_a_long_cycle_before_the_runtime_finishes(
    tmp_path,
) -> None:
    fake_runtime = tmp_path / "repeating_runtime.py"
    fake_runtime.write_text(
        """
import sys

cycle = " ".join(
    f"Corinth candidate {index} has distinct qualifier {index * 17}."
    for index in range(96)
)
print("ready")
print("you> ", end="", flush=True)
sys.stdin.readline()
print("llm> ", end="", flush=True)
print(cycle * 4 + cycle[:311], flush=True)
""".lstrip()
    )

    with pytest.raises(
        ConversationGateError,
        match="entered a repeated segment before termination",
    ):
        run_resident_conversation([sys.executable, "-u", str(fake_runtime)])


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
    reservation_probe = _reservation_probe()

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
            device_reservation_probe=reservation_probe,
        )

    transcript = (transcript_dir / "conversation-seed-19-failed.log").read_text()
    assert "failure-prefix" in transcript
    assert "precise terminal failure" in transcript
    assert "x" * 5000 in transcript
    assert reservation_probe.capture_count == 2


def test_gate_discards_one_complete_resident_conversation_and_measures_the_next(
    tmp_path,
) -> None:
    package = tmp_path / "package.json"
    package.write_text('{"package_id":"fixture"}')
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
print("nerve chat ready: physical_execution=VulkanMountedPhysicalExecutionSummary { tensor_parallel_island_count: 1, whole_expert_parallel_island_count: 2, intra_expert_tensor_parallel_island_count: 3, hybrid_island_count: 4, selected_resource_placement_count: 5 }, setup_ms=12.000")
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
    print("  distributed_decode_island_submissions=3")
    print("  distributed_decode_shard_submissions=6")
    print("  distributed_decode_tensor_parallel_island_submissions=1")
    print("  distributed_decode_whole_expert_parallel_island_submissions=1")
    print("  distributed_decode_intra_expert_tensor_parallel_island_submissions=1")
    print("  distributed_decode_hybrid_island_submissions=0")
    print("  distributed_prefill_island_submissions=3")
    print("  distributed_prefill_shard_submissions=6")
    print("  distributed_prefill_tensor_parallel_island_submissions=1")
    print("  distributed_prefill_whole_expert_parallel_island_submissions=1")
    print("  distributed_prefill_intra_expert_tensor_parallel_island_submissions=1")
    print("  distributed_prefill_hybrid_island_submissions=0")
    print("resource_residency:")
    print("  policy=demand-paged physical_stores=3")
    print(f"  payload_bytes(initial/current/high_water/maximum)=0/{misses * 10}/{misses * 10}/100 units(initial/current/high_water/addressable)=0/{misses}/{misses}/10")
    print(f"  gpu_accesses(selections/resident_hits/misses)={completed}/{completed - misses}/{misses}")
    print(f"  residency_requests(directory_hits/load_required/deduplicated/succeeded/failed/cancelled)={misses}/{misses}/0/{misses}/0/0")
    print("  residency_eviction(cycles/units/payload_bytes/device_bytes/reloads)=0/0/0/0/0")
    print(f"  memory_tiers(device_payload/host_visible_payload/device_capacity/host_visible_capacity)={misses * 8}/{misses * 2}/80/20")
    print(f"  transfers(reads/source_bytes/resident_bytes/uploaded_bytes/read_ms/derivation_ms/upload_ms/blocking_ms)={misses}/{misses * 10}/{misses * 20}/{misses * 20}/{misses}.0/{misses / 2}/{misses}.0/{misses}.0")
    print("determinism:")
    print(f"  generated_tokens=nerve.runtime.token_ids_sha256.v1:{conversation_turn:064x}")
    print(f"  selection_counters=nerve.runtime.selection_counters_sha256.v1:{conversation_turn + 100:064x}")
    print(f"  resident_state=nerve.optimizer.artifact_sha256.v1:{conversation_turn + 200:064x}", flush=True)
print("shutdown:")
print("  complete=true streams=1 packages=1 scheduler_in_flight=0")
print("  physical_devices_acknowledged=1/1 released_units=6 released_payload_bytes=60 cancelled_loads=0")
print("  package=fixture scope=target physical_devices_acknowledged=1/1")
print("  store=store0 physical_device=gpu0 acknowledged=true remaining_units=0 remaining_payload_bytes=0 error=None", flush=True)
""".lstrip()
        + 'print("device_restoration:")\n'
        + f"print({('  ' + json.dumps(_device_restoration_payload(), separators=(',', ':')))!r}, flush=True)\n"
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
        minimum_tensor_parallel_islands=8,
        require_thinking=True,
        warmup_conversation_sets=1,
        device_reservation_probe=_reservation_probe(),
    )

    run = report.runs[0]
    assert len(run.discarded_warmup_sets) == 2
    assert run.physical_execution is not None
    assert run.physical_execution.total_tensor_parallel_island_count == 8
    assert run.device_restoration.complete
    assert run.device_restoration.physical_device_count == 1
    assert run.device_restoration.devices[0].physical_device_id == "gpu0"
    assert run.external_device_reservation.complete
    assert run.external_device_reservation.selected_device_count == 1
    assert report.warmup_conversation_sets == 2
    assert run.discarded_warmup_sets[0].mean_decode_tokens_per_second == 1.0
    assert run.measured_set.mean_decode_tokens_per_second == 30.0
    assert run.measured_set.residency_policy == "demand-paged"
    assert run.measured_set.residency_delta is not None
    assert run.measured_set.residency_delta["gpu_accesses.selections"] == 6
    assert run.measured_set.residency_delta["gpu_accesses.misses"] == 0
    assert run.measured_set.residency_delta["transfers.blocking_ms"] == 0.0
    assert run.measured_set.residency_gauges_start == {
        "payload_bytes.initial": 0,
        "payload_bytes.current": 60,
        "payload_bytes.high_water": 60,
        "payload_bytes.maximum": 100,
        "units.initial": 0,
        "units.current": 6,
        "units.high_water": 6,
        "units.addressable": 10,
        "memory_tiers.device_payload": 48,
        "memory_tiers.host_visible_payload": 12,
        "memory_tiers.device_capacity": 80,
        "memory_tiers.host_visible_capacity": 20,
    }
    assert run.measured_set.residency_gauges_end == (
        run.measured_set.residency_gauges_start
    )


def test_gate_requires_two_consecutive_fully_warm_sets_after_recurrent_loads(
    tmp_path,
) -> None:
    package = tmp_path / "package.json"
    package.write_text('{"package_id":"fixture"}')
    fake_runtime = tmp_path / "recurrent_load_runtime.py"
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
loads_per_set = (6, 0, 1, 0, 0)
print("ready")
conversation_set = 0
conversation_turn = 0
while True:
    print("you> ", end="", flush=True)
    prompt = sys.stdin.readline().rstrip("\\n")
    if prompt == "/exit":
        break
    if prompt == "/new":
        if conversation_turn != len(answers):
            raise SystemExit(91)
        conversation_set += 1
        conversation_turn = 0
        print("session_reset: zeroed_state_buffers=17", flush=True)
        continue
    if conversation_set >= len(loads_per_set):
        raise SystemExit(92)
    answer = answers[conversation_turn]
    conversation_turn += 1
    prior_loads = sum(loads_per_set[:conversation_set])
    set_loads = min(conversation_turn, loads_per_set[conversation_set])
    loads = prior_loads + set_loads
    print(f"llm> reasoning</think> {answer}")
    print("stats:")
    print("  prefill_tokens_per_second=100.000")
    print(f"  decode_tokens_per_second={20 + conversation_set:.3f}")
    print("execution:")
    print("resource_residency:")
    print("  policy=demand-paged physical_stores=3")
    print(f"  payload_bytes(initial/current/high_water/maximum)=0/{loads * 10}/{loads * 10}/100 units(initial/current/high_water/addressable)=0/{loads}/{loads}/10")
    print(f"  gpu_accesses(selections/resident_hits/misses)={conversation_set * 6 + conversation_turn}/{conversation_set * 6 + conversation_turn - loads}/{loads}")
    print(f"  residency_requests(directory_hits/load_required/deduplicated/succeeded/failed/cancelled)={loads}/{loads}/0/{loads}/0/0")
    print("  residency_eviction(cycles/units/payload_bytes/device_bytes/reloads)=0/0/0/0/0")
    print(f"  memory_tiers(device_payload/host_visible_payload/device_capacity/host_visible_capacity)={loads * 8}/{loads * 2}/80/20")
    print(f"  transfers(reads/source_bytes/resident_bytes/uploaded_bytes/read_ms/derivation_ms/upload_ms/blocking_ms)={loads}/{loads * 10}/{loads * 20}/{loads * 20}/{loads}.0/{loads / 2}/{loads}.0/{loads}.0")
    print("determinism:")
    print(f"  generated_tokens=nerve.runtime.token_ids_sha256.v1:{conversation_turn:064x}")
    print(f"  selection_counters=nerve.runtime.selection_counters_sha256.v1:{conversation_turn + 100:064x}")
    print(f"  resident_state=nerve.optimizer.artifact_sha256.v1:{conversation_turn + 200:064x}", flush=True)
print("shutdown:")
print("  complete=true streams=1 packages=1 scheduler_in_flight=0")
print("  physical_devices_acknowledged=1/1 released_units=7 released_payload_bytes=70 cancelled_loads=0")
print("  package=fixture scope=target physical_devices_acknowledged=1/1")
print("  store=store0 physical_device=gpu0 acknowledged=true remaining_units=0 remaining_payload_bytes=0 error=None", flush=True)
""".lstrip()
        + 'print("device_restoration:")\n'
        + f"print({('  ' + json.dumps(_device_restoration_payload(), separators=(',', ':')))!r}, flush=True)\n"
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
        device_reservation_probe=_reservation_probe(),
    )

    run = report.runs[0]
    assert len(run.discarded_warmup_sets) == 4
    assert report.warmup_conversation_sets == 4
    assert [
        warmup.residency_delta["residency_requests.load_required"]
        for warmup in run.discarded_warmup_sets
        if warmup.residency_delta is not None
    ] == [6, 0, 1, 0]
    assert run.measured_set.residency_delta is not None
    assert run.measured_set.residency_delta["residency_requests.load_required"] == 0
    assert run.measured_set.mean_decode_tokens_per_second == 24.0


def _fully_warm_mock_transcript(*, mutation: str | None = None) -> str:
    answers = (
        "Hello.",
        "I am a language model.",
        "The capital of Greece is Athens.",
        "There are several cities named Corinth.",
        "My knowledge cutoff is unavailable.",
        "The country was Greece.",
    )
    sections = ["ready\n"]
    for conversation_set in range(2):
        for turn_index, answer in enumerate(answers, start=1):
            mutated = conversation_set == 1 and turn_index == 3
            response = _response(answer)
            digests = {
                "generated_tokens": (
                    f"nerve.runtime.token_ids_sha256.v1:{turn_index:064x}"
                ),
                "selection_counters": (
                    "nerve.runtime.selection_counters_sha256.v1:"
                    f"{turn_index + 100:064x}"
                ),
                "resident_state": (
                    f"nerve.optimizer.artifact_sha256.v1:{turn_index + 200:064x}"
                ),
            }
            if mutated and mutation in digests:
                prefix = digests[mutation].split(":", 1)[0]
                digests[mutation] = f"{prefix}:{999:064x}"
            elif mutated and mutation == "response":
                response = _response(answer + " This wording drifted.")
            sections.append(
                "you> llm> "
                f"{response}\n"
                "stats:\n"
                "  prefill_tokens_per_second=100.000\n"
                "  decode_tokens_per_second=25.000\n"
                "execution:\n"
                "resource_residency:\n"
                "  policy=demand-paged physical_stores=1\n"
                "  payload_bytes(initial/current/high_water/maximum)=0/8/8/80 units(initial/current/high_water/addressable)=0/1/1/10\n"
                f"  gpu_accesses(selections/resident_hits/misses)={turn_index}/{turn_index}/0\n"
                "  residency_requests(directory_hits/load_required/deduplicated/succeeded/failed/cancelled)=0/0/0/0/0/0\n"
                "  residency_eviction(cycles/units/payload_bytes/device_bytes/reloads)=0/0/0/0/0\n"
                "  memory_tiers(device_payload/host_visible_payload/device_capacity/host_visible_capacity)=8/0/80/20\n"
                "  transfers(reads/source_bytes/resident_bytes/uploaded_bytes/read_ms/derivation_ms/upload_ms/blocking_ms)=0/0/0/0/0/0/0/0\n"
            )
            if not (mutated and mutation == "missing_evidence"):
                sections.extend(
                    (
                        "determinism:\n",
                        f"  generated_tokens={digests['generated_tokens']}\n",
                        f"  selection_counters={digests['selection_counters']}\n",
                        f"  resident_state={digests['resident_state']}\n",
                    )
                )
    sections.append("you> ")
    sections.extend(
        (
            "shutdown:\n",
            "  complete=true streams=1 packages=1 scheduler_in_flight=0\n",
            "  physical_devices_acknowledged=1/1 released_units=1 released_payload_bytes=8 cancelled_loads=0\n",
            "  package=fixture scope=target physical_devices_acknowledged=1/1\n",
            "  store=store0 physical_device=gpu0 acknowledged=true remaining_units=0 remaining_payload_bytes=0 error=None\n",
        )
    )
    sections.append(_valid_device_restoration_transcript())
    return "".join(sections)


def test_gate_rejects_loss_of_preexisting_external_gpu_allocation(
    monkeypatch,
    tmp_path,
) -> None:
    package = tmp_path / "package.json"
    package.write_text('{"package_id":"fixture"}')
    process = DrmProcessReservation(
        pid=42,
        start_time_ticks=100,
        command="existing-model",
        vram_bytes=96 * 1024**2,
        shared_bytes=4 * 1024**2,
    )
    probe = _StaticDeviceReservationProbe(
        _external_reservation_snapshot(processes=(process,)),
        _external_reservation_snapshot(processes=()),
    )
    monkeypatch.setattr(
        "nerve.conversation_gate.run_resident_conversation",
        lambda command, **_kwargs: (_fully_warm_mock_transcript(), 0),
    )

    with pytest.raises(
        ConversationGateError,
        match="lost pre-existing process",
    ):
        run_conversation_gate(
            [sys.executable, "--package", str(package), "--chat"],
            seeds=(0,),
            minimum_decode_tokens_per_second=20.0,
            require_thinking=True,
            warmup_conversation_sets=1,
            device_reservation_probe=probe,
        )


@pytest.mark.parametrize(
    ("replace", "message"),
    (
        (
            (f"generated_tokens=nerve.runtime.token_ids_sha256.v1:{1:064x}"),
            "not a canonical digest",
        ),
        ("generated_tokens=", "unknown, duplicate, or empty field"),
    ),
)
def test_transcript_parser_rejects_untrusted_determinism_evidence(
    replace: str,
    message: str,
) -> None:
    transcript = _fully_warm_mock_transcript()
    if replace == "generated_tokens=":
        transcript = transcript.replace(replace, "unknown_digest=", 1)
    else:
        transcript = transcript.replace(replace, "generated_tokens=not-a-digest", 1)

    with pytest.raises(ConversationGateError, match=message):
        parse_conversation_transcript(transcript, warmup_conversation_sets=1)


def test_transcript_parser_rejects_duplicate_determinism_evidence() -> None:
    transcript = _fully_warm_mock_transcript()
    evidence = (
        "determinism:\n"
        f"  generated_tokens=nerve.runtime.token_ids_sha256.v1:{1:064x}\n"
        f"  selection_counters=nerve.runtime.selection_counters_sha256.v1:{101:064x}\n"
        f"  resident_state=nerve.optimizer.artifact_sha256.v1:{201:064x}\n"
    )
    transcript = transcript.replace(evidence, evidence + evidence, 1)

    with pytest.raises(ConversationGateError, match="more than one determinism"):
        parse_conversation_transcript(transcript, warmup_conversation_sets=1)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("generated_tokens", "turn 3 changed its generated tokens digest"),
        ("selection_counters", "turn 3 changed its selection counters digest"),
        ("resident_state", "turn 3 changed its resident state digest"),
        ("response", "turn 3 changed its decoded response"),
        ("missing_evidence", "turn 3 lacks determinism evidence"),
    ),
)
def test_gate_rejects_fully_warm_behavior_or_state_drift(
    monkeypatch,
    tmp_path,
    mutation: str,
    message: str,
) -> None:
    package = tmp_path / "package.json"
    package.write_text('{"package_id":"fixture"}')
    monkeypatch.setattr(
        "nerve.conversation_gate.run_resident_conversation",
        lambda command, **_kwargs: (
            _fully_warm_mock_transcript(mutation=mutation),
            0,
        ),
    )

    with pytest.raises(
        ConversationGateError,
        match=message,
    ):
        run_conversation_gate(
            [sys.executable, "--package", str(package), "--chat"],
            seeds=(0,),
            minimum_decode_tokens_per_second=20.0,
            require_thinking=True,
            warmup_conversation_sets=1,
            device_reservation_probe=_reservation_probe(),
        )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        (
            "released_units=1 released_payload_bytes=8",
            "released_units=2 released_payload_bytes=8",
            "released units do not match final residency",
        ),
        (
            "released_units=1 released_payload_bytes=8",
            "released_units=1 released_payload_bytes=9",
            "released payload bytes do not match final residency",
        ),
        (
            "package=fixture scope=target",
            "package=another-package scope=target",
            "package identity does not match",
        ),
    ),
)
def test_gate_reconciles_shutdown_with_package_and_final_residency(
    monkeypatch,
    tmp_path,
    old: str,
    new: str,
    message: str,
) -> None:
    package = tmp_path / "package.json"
    package.write_text('{"package_id":"fixture"}')
    transcript = _fully_warm_mock_transcript().replace(old, new, 1)
    monkeypatch.setattr(
        "nerve.conversation_gate.run_resident_conversation",
        lambda command, **_kwargs: (transcript, 0),
    )

    with pytest.raises(ConversationGateError, match=message):
        run_conversation_gate(
            [sys.executable, "--package", str(package), "--chat"],
            seeds=(0,),
            minimum_decode_tokens_per_second=20.0,
            require_thinking=True,
            warmup_conversation_sets=1,
            device_reservation_probe=_reservation_probe(),
        )


def test_gate_requires_residency_evidence_for_shutdown_reconciliation(
    monkeypatch,
    tmp_path,
) -> None:
    package = tmp_path / "package.json"
    package.write_text('{"package_id":"fixture"}')
    transcript = _fully_warm_mock_transcript().replace(
        "  policy=demand-paged physical_stores=1\n", ""
    )
    monkeypatch.setattr(
        "nerve.conversation_gate.run_resident_conversation",
        lambda command, **_kwargs: (transcript, 0),
    )

    with pytest.raises(
        ConversationGateError, match="did not report resource residency"
    ):
        run_conversation_gate(
            [sys.executable, "--package", str(package), "--chat"],
            seeds=(0,),
            minimum_decode_tokens_per_second=20.0,
            require_thinking=True,
            warmup_conversation_sets=1,
            device_reservation_probe=_reservation_probe(),
        )


def test_resident_runner_fails_when_full_residency_never_stabilizes(tmp_path) -> None:
    fake_runtime = tmp_path / "never_warm_runtime.py"
    fake_runtime.write_text(
        """
import sys

print("ready")
completed = 0
conversation_turn = 0
while True:
    print("you> ", end="", flush=True)
    prompt = sys.stdin.readline().rstrip("\\n")
    if prompt == "/exit":
        break
    if prompt == "/new":
        conversation_turn = 0
        print("session_reset: zeroed_state_buffers=1", flush=True)
        continue
    completed += 1
    conversation_turn += 1
    print("llm> reasoning</think> answer")
    print("stats:")
    print("  prefill_tokens_per_second=100.000")
    print("  decode_tokens_per_second=25.000")
    print("execution:")
    print("resource_residency:")
    print("  policy=demand-paged physical_stores=1")
    print(f"  gpu_accesses(selections/resident_hits/misses)={completed}/0/{completed}")
    print(f"  residency_requests(directory_hits/load_required/deduplicated/succeeded/failed/cancelled)={completed}/{completed}/0/{completed}/0/0")
    print("  residency_eviction(cycles/units/payload_bytes/device_bytes/reloads)=0/0/0/0/0")
    print(f"  memory_tiers(device_payload/host_visible_payload/device_capacity/host_visible_capacity)={completed}/0/80/20")
    print(f"  transfers(reads/source_bytes/resident_bytes/uploaded_bytes/read_ms/derivation_ms/upload_ms/blocking_ms)={completed}/{completed}/{completed}/{completed}/{completed}.0/0.0/0.0/{completed}.0", flush=True)
""".lstrip()
    )

    with pytest.raises(
        ConversationGateError,
        match="did not produce two consecutive fully warm conversation sets within 3 sets",
    ):
        run_resident_conversation(
            [sys.executable, "-u", str(fake_runtime)],
            warmup_conversation_sets=1,
            warm_until_fully_resident=True,
            maximum_conversation_sets=3,
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

    with pytest.raises(ConversationGateError, match="repeated segment"):
        run_resident_conversation([sys.executable, "-u", str(fake_runtime)])
