from __future__ import annotations

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.validation.contracts import ValidationRun


_AMPLIFICATION_COUNTERS = (
    "bounded_wait_count",
    "bounded_wait_timeout_count",
    "execution_quantum_forced_yield_count",
    "resident_copy_waits",
    "resident_sequence_fence_waits",
)
_TRANSFER_COUNTERS = (
    "direct_copy_byte_count",
    "direct_copy_count",
    "direct_receive_byte_count",
    "direct_receive_count",
    "published_byte_count",
    "published_packet_count",
    "received_byte_count",
    "received_packet_count",
)


def qualify_whole_model_product_performance(
    run: ValidationRun | None,
) -> Json:
    if run is None:
        return {
            "status": "not_run",
            "reason": (
                "whole-model product performance was not measured because "
                "an earlier qualification gate failed"
            ),
            "metrics": {},
        }
    document = run.to_json()
    if document["stage"] != "whole_model":
        raise ModelCompileError(
            "product performance requires a whole-model validation run"
        )
    if document["status"] != "completed":
        return {
            "status": "not_run",
            "reason": (
                "whole-model product performance was not evaluated because "
                "behavioral execution did not complete"
            ),
            "metrics": {},
        }

    comparisons = tuple(
        _observation_comparison(observation)
        for observation in document["observations"]
    )
    if not comparisons:
        raise ModelCompileError(
            "whole-model product performance has no observations"
        )
    candidate_is_faster = all(
        comparison["candidate_measured_elapsed_ns"]
        < comparison["reference_measured_elapsed_ns"]
        for comparison in comparisons
    )
    amplified = sorted(
        {
            path
            for comparison in comparisons
            for path, delta in comparison["slow_path_deltas"].items()
            if delta > 0
        }
    )
    status = "passed" if candidate_is_faster and not amplified else "failed"
    reason = None
    if not candidate_is_faster:
        reason = (
            "candidate was not faster than the exact implementation in "
            "every warmed whole-model observation"
        )
    elif amplified:
        reason = (
            "candidate amplified one or more whole-model runtime slow paths: "
            + ", ".join(amplified)
        )
    return {
        "status": status,
        "reason": reason,
        "metrics": {
            "observation_count": len(comparisons),
            "warmup_turns_discarded": len(comparisons),
            "candidate_faster_observation_count": sum(
                comparison["candidate_measured_elapsed_ns"]
                < comparison["reference_measured_elapsed_ns"]
                for comparison in comparisons
            ),
            "reference_measured_elapsed_ns": sum(
                comparison["reference_measured_elapsed_ns"]
                for comparison in comparisons
            ),
            "candidate_measured_elapsed_ns": sum(
                comparison["candidate_measured_elapsed_ns"]
                for comparison in comparisons
            ),
            "host_speedup_ppm": _speedup_ppm(
                reference=sum(
                    comparison["reference_measured_elapsed_ns"]
                    for comparison in comparisons
                ),
                candidate=sum(
                    comparison["candidate_measured_elapsed_ns"]
                    for comparison in comparisons
                ),
            ),
            "reference_generated_tokens": sum(
                comparison["reference_generated_tokens"]
                for comparison in comparisons
            ),
            "candidate_generated_tokens": sum(
                comparison["candidate_generated_tokens"]
                for comparison in comparisons
            ),
            "reference_speculative_acceptance_ppm": _acceptance_ppm(
                proposed=sum(
                    comparison["reference_speculative_proposed"]
                    for comparison in comparisons
                ),
                accepted=sum(
                    comparison["reference_speculative_accepted"]
                    for comparison in comparisons
                ),
            ),
            "candidate_speculative_acceptance_ppm": _acceptance_ppm(
                proposed=sum(
                    comparison["candidate_speculative_proposed"]
                    for comparison in comparisons
                ),
                accepted=sum(
                    comparison["candidate_speculative_accepted"]
                    for comparison in comparisons
                ),
            ),
            "amplified_slow_paths": amplified,
            "slow_path_deltas": _sum_deltas(comparisons),
            "observations": list(comparisons),
        },
    }


def _observation_comparison(observation: Json) -> Json:
    statistics = observation.get("execution_statistics")
    if not isinstance(statistics, dict):
        raise ModelCompileError(
            "whole-model observation has no execution statistics"
        )
    roles = {
        role: _role_measurement(statistics.get(role), role=role)
        for role in ("reference", "candidate")
    }
    if roles["reference"]["generated_tokens"] != roles["candidate"][
        "generated_tokens"
    ]:
        raise ModelCompileError(
            "whole-model product timing compared different generated work"
        )
    slow_path_deltas = {
        path: roles["candidate"]["slow_paths"].get(path, 0)
        - roles["reference"]["slow_paths"].get(path, 0)
        for path in sorted(
            set(roles["reference"]["slow_paths"])
            | set(roles["candidate"]["slow_paths"])
        )
    }
    return {
        "observation_id": observation["observation_id"],
        "reference_measured_elapsed_ns": roles["reference"]["elapsed_ns"],
        "candidate_measured_elapsed_ns": roles["candidate"]["elapsed_ns"],
        "host_speedup_ppm": _speedup_ppm(
            reference=roles["reference"]["elapsed_ns"],
            candidate=roles["candidate"]["elapsed_ns"],
        ),
        "reference_generated_tokens": roles["reference"]["generated_tokens"],
        "candidate_generated_tokens": roles["candidate"]["generated_tokens"],
        "reference_speculative_proposed": roles["reference"][
            "speculative_proposed"
        ],
        "reference_speculative_accepted": roles["reference"][
            "speculative_accepted"
        ],
        "candidate_speculative_proposed": roles["candidate"][
            "speculative_proposed"
        ],
        "candidate_speculative_accepted": roles["candidate"][
            "speculative_accepted"
        ],
        "slow_path_deltas": slow_path_deltas,
    }


def _role_measurement(value: object, *, role: str) -> Json:
    if not isinstance(value, dict):
        raise ModelCompileError(
            f"whole-model {role} execution statistics are missing"
        )
    turns = value.get("turn_statistics")
    if (
        not isinstance(turns, list)
        or len(turns) < 2
        or any(not isinstance(turn, dict) for turn in turns)
    ):
        raise ModelCompileError(
            "whole-model product timing requires one discarded warmup "
            "followed by one or more measured turns"
        )
    indices = [_integer(turn.get("turn_index"), "turn_index") for turn in turns]
    if indices != list(range(len(turns))):
        raise ModelCompileError(
            "whole-model product timing turns are not contiguous"
        )
    measured = turns[1:]
    elapsed_ns = sum(
        _positive_integer(turn.get("elapsed_ns"), "turn elapsed_ns")
        for turn in measured
    )
    generated_tokens = sum(
        _positive_integer(
            turn.get("generated_tokens"),
            "turn generated_tokens",
        )
        for turn in measured
    )
    proposed = sum(
        _nonnegative_integer(
            _object(turn.get("speculative"), "turn speculative").get(
                "proposed_draft_tokens"
            ),
            "turn speculative.proposed_draft_tokens",
        )
        for turn in measured
    )
    accepted = sum(
        _nonnegative_integer(
            _object(turn.get("speculative"), "turn speculative").get(
                "accepted_draft_tokens"
            ),
            "turn speculative.accepted_draft_tokens",
        )
        for turn in measured
    )
    if accepted > proposed:
        raise ModelCompileError(
            "whole-model speculative acceptance exceeds proposals"
        )
    slow_paths: dict[str, int] = {}
    for turn in measured:
        counters = _object(
            turn.get("execution_counters"),
            "turn execution_counters",
        )
        feedback = _object(
            turn.get("resident_feedback"),
            "turn resident_feedback",
        )
        transport = _object(
            turn.get("transport"),
            "turn transport",
        )
        for name in _AMPLIFICATION_COUNTERS:
            source = feedback if name.startswith("bounded_wait") else counters
            if name in source:
                slow_paths[name] = slow_paths.get(name, 0) + _nonnegative_integer(
                    source[name],
                    name,
                )
        for name in _TRANSFER_COUNTERS:
            if name in transport:
                path = f"transport.{name}"
                slow_paths[path] = slow_paths.get(path, 0) + _nonnegative_integer(
                    transport[name],
                    path,
                )
    return {
        "elapsed_ns": elapsed_ns,
        "generated_tokens": generated_tokens,
        "speculative_proposed": proposed,
        "speculative_accepted": accepted,
        "slow_paths": slow_paths,
    }


def _sum_deltas(comparisons: tuple[Json, ...]) -> Json:
    paths = sorted(
        {
            path
            for comparison in comparisons
            for path in comparison["slow_path_deltas"]
        }
    )
    return {
        path: sum(
            comparison["slow_path_deltas"].get(path, 0)
            for comparison in comparisons
        )
        for path in paths
    }


def _speedup_ppm(*, reference: int, candidate: int) -> int:
    return (reference - candidate) * 1_000_000 // reference


def _acceptance_ppm(*, proposed: int, accepted: int) -> int | None:
    return None if proposed == 0 else accepted * 1_000_000 // proposed


def _object(value: object, path: str) -> Json:
    if not isinstance(value, dict):
        raise ModelCompileError(f"{path} must be an object")
    return value


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelCompileError(f"{path} must be an integer")
    return value


def _nonnegative_integer(value: object, path: str) -> int:
    value = _integer(value, path)
    if value < 0:
        raise ModelCompileError(f"{path} must not be negative")
    return value


def _positive_integer(value: object, path: str) -> int:
    value = _integer(value, path)
    if value <= 0:
        raise ModelCompileError(f"{path} must be positive")
    return value
