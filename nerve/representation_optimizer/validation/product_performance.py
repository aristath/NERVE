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
    *,
    product_check_ids: frozenset[str],
) -> Json:
    if not product_check_ids:
        raise ModelCompileError(
            "whole-model product performance has no declared check"
        )
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

    product_observations = tuple(
        observation
        for observation in document["observations"]
        if observation["check_id"] in product_check_ids
    )
    observed_check_ids = {
        observation["check_id"]
        for observation in product_observations
    }
    if observed_check_ids != product_check_ids:
        raise ModelCompileError(
            "whole-model product performance did not execute every "
            "declared product check"
        )
    comparisons = tuple(
        _observation_comparison(observation)
        for observation in product_observations
    )
    if not comparisons:
        raise ModelCompileError(
            "whole-model product performance has no observations"
        )
    candidate_is_faster = all(
        comparison["candidate_normalized_faster"]
        for comparison in comparisons
    )
    amplified = sorted(
        {
            path
            for comparison in comparisons
            for path in comparison["amplified_runtime_paths"]
        }
    )
    status = "passed" if candidate_is_faster and not amplified else "failed"
    reasons = []
    if not candidate_is_faster:
        reasons.append(
            "candidate was not faster than the exact implementation after "
            "normalizing warmed elapsed time by generated tokens"
        )
    if amplified:
        reasons.append(
            "candidate amplified one or more normalized whole-model runtime "
            "slow paths: "
            + ", ".join(amplified)
        )
    reference_elapsed = sum(
        comparison["reference_measured_elapsed_ns"]
        for comparison in comparisons
    )
    candidate_elapsed = sum(
        comparison["candidate_measured_elapsed_ns"]
        for comparison in comparisons
    )
    reference_tokens = sum(
        comparison["reference_generated_tokens"]
        for comparison in comparisons
    )
    candidate_tokens = sum(
        comparison["candidate_generated_tokens"]
        for comparison in comparisons
    )
    slow_paths = _aggregate_slow_paths(comparisons)
    return {
        "status": status,
        "reason": "; ".join(reasons) or None,
        "metrics": {
            "observation_count": len(comparisons),
            "warmup_turns_discarded": len(comparisons),
            "candidate_faster_observation_count": sum(
                comparison["candidate_normalized_faster"]
                for comparison in comparisons
            ),
            "reference_measured_elapsed_ns": reference_elapsed,
            "candidate_measured_elapsed_ns": candidate_elapsed,
            "host_speedup_ppm": _normalized_speedup_ppm(
                reference_elapsed_ns=reference_elapsed,
                reference_tokens=reference_tokens,
                candidate_elapsed_ns=candidate_elapsed,
                candidate_tokens=candidate_tokens,
            ),
            "reference_generated_tokens": reference_tokens,
            "candidate_generated_tokens": candidate_tokens,
            "reference_ns_per_generated_token": (
                reference_elapsed // reference_tokens
            ),
            "candidate_ns_per_generated_token": (
                candidate_elapsed // candidate_tokens
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
            "slow_path_count_deltas": slow_paths["count_deltas"],
            "slow_path_rate_deltas_per_million_tokens": slow_paths[
                "rate_deltas_per_million_tokens"
            ],
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
    slow_path_count_deltas = {
        path: roles["candidate"]["slow_paths"].get(path, 0)
        - roles["reference"]["slow_paths"].get(path, 0)
        for path in sorted(
            set(roles["reference"]["slow_paths"])
            | set(roles["candidate"]["slow_paths"])
        )
    }
    slow_path_rate_deltas = {
        path: _rate_per_million(
            roles["candidate"]["slow_paths"].get(path, 0),
            roles["candidate"]["generated_tokens"],
        )
        - _rate_per_million(
            roles["reference"]["slow_paths"].get(path, 0),
            roles["reference"]["generated_tokens"],
        )
        for path in slow_path_count_deltas
    }
    amplified_runtime_paths = {
        path
        for path in slow_path_count_deltas
        if _rate_is_greater(
            candidate_count=roles["candidate"]["slow_paths"].get(path, 0),
            candidate_tokens=roles["candidate"]["generated_tokens"],
            reference_count=roles["reference"]["slow_paths"].get(path, 0),
            reference_tokens=roles["reference"]["generated_tokens"],
        )
    }
    reference_acceptance = _acceptance_ppm(
        proposed=roles["reference"]["speculative_proposed"],
        accepted=roles["reference"]["speculative_accepted"],
    )
    candidate_acceptance = _acceptance_ppm(
        proposed=roles["candidate"]["speculative_proposed"],
        accepted=roles["candidate"]["speculative_accepted"],
    )
    if (
        reference_acceptance is not None
        and (
            candidate_acceptance is None
            or candidate_acceptance < reference_acceptance
        )
    ):
        amplified_runtime_paths.add("speculative.acceptance")
    return {
        "observation_id": observation["observation_id"],
        "reference_measured_elapsed_ns": roles["reference"]["elapsed_ns"],
        "candidate_measured_elapsed_ns": roles["candidate"]["elapsed_ns"],
        "candidate_normalized_faster": _normalized_faster(
            reference_elapsed_ns=roles["reference"]["elapsed_ns"],
            reference_tokens=roles["reference"]["generated_tokens"],
            candidate_elapsed_ns=roles["candidate"]["elapsed_ns"],
            candidate_tokens=roles["candidate"]["generated_tokens"],
        ),
        "host_speedup_ppm": _normalized_speedup_ppm(
            reference_elapsed_ns=roles["reference"]["elapsed_ns"],
            reference_tokens=roles["reference"]["generated_tokens"],
            candidate_elapsed_ns=roles["candidate"]["elapsed_ns"],
            candidate_tokens=roles["candidate"]["generated_tokens"],
        ),
        "reference_generated_tokens": roles["reference"]["generated_tokens"],
        "candidate_generated_tokens": roles["candidate"]["generated_tokens"],
        "reference_ns_per_generated_token": (
            roles["reference"]["elapsed_ns"]
            // roles["reference"]["generated_tokens"]
        ),
        "candidate_ns_per_generated_token": (
            roles["candidate"]["elapsed_ns"]
            // roles["candidate"]["generated_tokens"]
        ),
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
        "reference_speculative_acceptance_ppm": reference_acceptance,
        "candidate_speculative_acceptance_ppm": candidate_acceptance,
        "reference_slow_paths": roles["reference"]["slow_paths"],
        "candidate_slow_paths": roles["candidate"]["slow_paths"],
        "slow_path_count_deltas": slow_path_count_deltas,
        "slow_path_rate_deltas_per_million_tokens": (
            slow_path_rate_deltas
        ),
        "amplified_runtime_paths": sorted(amplified_runtime_paths),
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


def _aggregate_slow_paths(comparisons: tuple[Json, ...]) -> Json:
    paths = sorted(
        {
            path
            for comparison in comparisons
            for role in ("reference", "candidate")
            for path in comparison[f"{role}_slow_paths"]
        }
    )
    reference_tokens = sum(
        comparison["reference_generated_tokens"]
        for comparison in comparisons
    )
    candidate_tokens = sum(
        comparison["candidate_generated_tokens"]
        for comparison in comparisons
    )
    reference_counts = {
        path: sum(
            comparison["reference_slow_paths"].get(path, 0)
            for comparison in comparisons
        )
        for path in paths
    }
    candidate_counts = {
        path: sum(
            comparison["candidate_slow_paths"].get(path, 0)
            for comparison in comparisons
        )
        for path in paths
    }
    return {
        "count_deltas": {
            path: candidate_counts[path] - reference_counts[path]
            for path in paths
        },
        "rate_deltas_per_million_tokens": {
            path: (
                _rate_per_million(
                    candidate_counts[path],
                    candidate_tokens,
                )
                - _rate_per_million(
                    reference_counts[path],
                    reference_tokens,
                )
            )
            for path in paths
        },
    }


def _normalized_faster(
    *,
    reference_elapsed_ns: int,
    reference_tokens: int,
    candidate_elapsed_ns: int,
    candidate_tokens: int,
) -> bool:
    return (
        candidate_elapsed_ns * reference_tokens
        < reference_elapsed_ns * candidate_tokens
    )


def _normalized_speedup_ppm(
    *,
    reference_elapsed_ns: int,
    reference_tokens: int,
    candidate_elapsed_ns: int,
    candidate_tokens: int,
) -> int:
    reference_scaled = reference_elapsed_ns * candidate_tokens
    candidate_scaled = candidate_elapsed_ns * reference_tokens
    return (
        (reference_scaled - candidate_scaled)
        * 1_000_000
        // reference_scaled
    )


def _rate_per_million(count: int, tokens: int) -> int:
    return count * 1_000_000 // tokens


def _rate_is_greater(
    *,
    candidate_count: int,
    candidate_tokens: int,
    reference_count: int,
    reference_tokens: int,
) -> bool:
    return (
        candidate_count * reference_tokens
        > reference_count * candidate_tokens
    )


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
