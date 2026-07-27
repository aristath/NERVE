from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.benchmarking.contracts import (
    BENCHMARK_PLAN_SCHEMA,
    BENCHMARK_WORKLOAD_SCHEMA,
    BenchmarkPlan,
    BenchmarkWorkload,
    benchmark_plan_id,
    benchmark_workload_id,
)
from nerve.representation_optimizer.contracts import (
    CANDIDATE_CONSTRUCTION_SCHEMA,
    HARDWARE_PROCESS_PROFILE_SCHEMA,
    ContractDocument,
    contract_digest,
    validate_contract,
)
from nerve.representation_optimizer.providers.types import ProviderCandidatePlan


@dataclass(frozen=True)
class BenchmarkPolicy:
    minimum_warmup_samples: int = 4
    maximum_warmup_samples: int = 12
    warmup_stability_window_samples: int = 2
    measured_pairs_per_block: int = 3
    minimum_measured_pairs_per_seed: int = 6
    maximum_measured_pairs_per_seed: int = 30
    confidence_level_ppm: int = 950_000
    minimum_material_improvement_ppm: int = 50_000
    maximum_relative_ci_width_ppm: int = 100_000
    maximum_order_bias_ppm: int = 75_000
    maximum_warmup_shift_ppm: int = 50_000
    maximum_sustained_regression_ppm: int = 50_000

    def to_json(self) -> Json:
        return asdict(self)


def create_benchmark_workload(
    *,
    name: str,
    execution_phase: str,
    activation_batch_width: int,
    context_size: int,
    state_size: int,
    stream_count: int,
    mount_mode: str,
    boundary_mode: str,
    input_artifact: Json,
    initial_state_artifact: Json | None,
    controls: Json,
    randomness_algorithm: str,
    seeds: Iterable[int],
    deterministic_replay_required: bool,
    permit_sampling_variance: bool,
    permit_numerical_nondeterminism: bool,
    permit_speculative_schedule_variance: bool,
    useful_work_unit: str,
    minimum_useful_work_units: int,
    completion_condition: str,
    output_allowance: int | None,
    output_allowance_basis: Json,
    sustained_window_count: int,
) -> BenchmarkWorkload:
    document = {
        "schema": BENCHMARK_WORKLOAD_SCHEMA,
        "workload_id": "",
        "name": name,
        "regime": {
            "execution_phase": execution_phase,
            "activation_batch_width": activation_batch_width,
            "context_size": context_size,
            "state_size": state_size,
            "stream_count": stream_count,
            "mount_mode": mount_mode,
            "boundary_mode": boundary_mode,
        },
        "input": input_artifact,
        "initial_state": initial_state_artifact,
        "controls": controls,
        "randomness": {
            "algorithm": randomness_algorithm,
            "seeds": sorted(set(seeds)),
            "deterministic_replay_required": deterministic_replay_required,
            "permit_sampling_variance": permit_sampling_variance,
            "permit_numerical_nondeterminism": (permit_numerical_nondeterminism),
            "permit_speculative_schedule_variance": (
                permit_speculative_schedule_variance
            ),
        },
        "useful_work": {
            "unit": useful_work_unit,
            "minimum_units": minimum_useful_work_units,
            "completion_condition": completion_condition,
            "output_allowance": output_allowance,
            "output_allowance_basis": dict(output_allowance_basis),
            "matched_work_policy": "equal_useful_work",
            "sustained_window_count": sustained_window_count,
        },
    }
    document["workload_id"] = benchmark_workload_id(document)
    return BenchmarkWorkload.from_json(document)


def build_benchmark_plan(
    *,
    candidate_plan: ProviderCandidatePlan,
    construction_record: ContractDocument,
    hardware_profiles: Iterable[Json],
    reference_implementation_id: str,
    reference_contract_digest: str,
    reference_artifact_refs: Iterable[Json],
    matched_conditions: Json,
    policy: BenchmarkPolicy | None = None,
) -> BenchmarkPlan:
    construction = construction_record.to_json()
    if (
        construction["schema"] != CANDIDATE_CONSTRUCTION_SCHEMA
        or construction["status"] != "completed"
        or construction["candidate_id"] != candidate_plan.candidate_id
    ):
        raise ModelCompileError(
            "benchmark planning requires the completed construction record "
            "for its candidate"
        )
    profiles = [dict(profile) for profile in hardware_profiles]
    if not profiles:
        raise ModelCompileError(
            "benchmark planning requires at least one hardware profile"
        )
    for profile in profiles:
        validate_contract(
            profile,
            expected_schema=HARDWARE_PROCESS_PROFILE_SCHEMA,
        )
    candidate = candidate_plan.candidate.to_json()
    capability_classes = {profile["capability_class"] for profile in profiles}
    target_capability = candidate["target_predicate"].get("capability_class")
    if target_capability is not None and target_capability not in capability_classes:
        raise ModelCompileError(
            "candidate target predicate does not match benchmark hardware"
        )
    expected_devices = sorted(
        (
            {
                "device_id": profile["hardware_identity"]["stable_device_id"],
                "hardware_profile_digest": contract_digest(profile),
                "capability_class": profile["capability_class"],
                "api": profile["provenance"]["api"],
            }
            for profile in profiles
        ),
        key=lambda device: device["device_id"],
    )
    if matched_conditions.get("devices") != expected_devices:
        raise ModelCompileError(
            "matched benchmark devices do not match supplied hardware profiles"
        )
    workloads = [
        workload.to_json()
        for workload in sorted(
            candidate_plan.benchmark_workloads,
            key=lambda workload: workload.workload_id,
        )
    ]
    if not workloads:
        raise ModelCompileError("candidate has no matched benchmark workloads")
    source_contract_digests = list(candidate["source_contract_digests"])
    if source_contract_digests != sorted(set(source_contract_digests)):
        raise ModelCompileError(
            "candidate source contracts must be sorted before benchmark planning"
        )
    if reference_contract_digest not in source_contract_digests:
        raise ModelCompileError(
            "exact reference implementation is not bound to a candidate "
            "source behavior contract"
        )
    candidate_artifacts = [
        {"path": artifact["path"], "digest": artifact["digest"]}
        for artifact in construction["artifacts"]
    ]
    implementations = {
        "reference": {
            "implementation_id": reference_implementation_id,
            "contract_digest": reference_contract_digest,
            "artifact_refs": sorted(
                (dict(reference) for reference in reference_artifact_refs),
                key=lambda reference: reference["path"],
            ),
        },
        "candidate": {
            "implementation_id": (
                f"staged-representation:{candidate_plan.candidate_id}"
            ),
            "contract_digest": construction_record.digest,
            "artifact_refs": candidate_artifacts,
        },
    }
    selected_policy = (policy or BenchmarkPolicy()).to_json()
    document = {
        "schema": BENCHMARK_PLAN_SCHEMA,
        "plan_id": "",
        "candidate_id": candidate_plan.candidate_id,
        "construction_record_digest": construction_record.digest,
        "source_contract_digests": source_contract_digests,
        "implementations": implementations,
        "matched_conditions": matched_conditions,
        "matched_conditions_digest": contract_digest(matched_conditions),
        "workloads": workloads,
        "policy": selected_policy,
    }
    document["plan_id"] = benchmark_plan_id(document)
    return BenchmarkPlan.from_json(document)
