from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from nerve.compilation import Json, ModelCompileError, read_json
from nerve.representation_optimizer.analysis.evidence import (
    AnalysisRun,
    validate_analysis_run_directory,
)
from nerve.representation_optimizer.benchmarking.storage import (
    load_benchmark_evidence,
)
from nerve.representation_optimizer.contracts import (
    BENCHMARK_RECORD_SCHEMA,
    CANDIDATE_CONSTRUCTION_SCHEMA,
    HARDWARE_PROCESS_PROFILE_SCHEMA,
    VALIDATION_RECORD_SCHEMA,
    ContractDocument,
    contract_digest,
    validate_contract,
)
from nerve.representation_optimizer.lifecycle import (
    CandidateState,
    OptimizationSession,
)
from nerve.representation_optimizer.promotion.contracts import (
    PROMOTION_DECISION_SCHEMA,
    PromotionDecision,
    RuntimeImplementationPredicate,
    create_runtime_implementation_predicate,
    implementation_id,
    promotion_decision_id,
)
from nerve.representation_optimizer.providers.types import (
    ProviderCandidatePlan,
)
from nerve.representation_optimizer.staging.integrity import (
    integrity_evidence,
    validate_staged_candidate,
)
from nerve.representation_optimizer.staging.loading import (
    LoadedStagedCandidate,
    load_staged_candidate,
)
from nerve.representation_optimizer.validation.storage import (
    load_prebenchmark_evidence,
    load_validation_evidence,
)


@dataclass(frozen=True)
class PreparedPromotion:
    candidate_plan: ProviderCandidatePlan
    construction_record: ContractDocument
    prebenchmark_record: ContractDocument
    benchmark_record: ContractDocument
    validation_record: ContractDocument
    runtime_predicate: RuntimeImplementationPredicate
    decision: PromotionDecision
    registry_entry: Json
    staged_candidate: LoadedStagedCandidate
    benchmark_evidence_path: Path
    prebenchmark_evidence_path: Path
    validation_evidence_path: Path
    analysis_runs: tuple[PreparedAnalysisRun, ...]
    hardware_profiles: tuple[Json, ...]
    session: OptimizationSession

    @property
    def implementation_id(self) -> str:
        return self.decision.implementation_id


@dataclass(frozen=True)
class PreparedAnalysisRun:
    path: Path
    run: AnalysisRun
    run_digest: str
    cited_evidence_ids: tuple[str, ...]


def prepare_candidate_promotion(
    *,
    package_dir: Path,
    candidate_workspace_root: Path,
    benchmark_workspace_root: Path,
    validation_workspace_root: Path,
    analysis_run_directories: Iterable[Path],
    candidate_plan: ProviderCandidatePlan,
    construction_record: ContractDocument,
    benchmark_record: ContractDocument,
    validation_record: ContractDocument,
    hardware_profiles: Iterable[Json],
    session: OptimizationSession,
    reason: str,
) -> PreparedPromotion:
    if not reason:
        raise ModelCompileError("promotion reason must not be empty")
    _require_candidate_state(
        session,
        candidate_plan.candidate_id,
        CandidateState.BEHAVIORALLY_VALIDATED,
    )
    _validate_record_headers(
        candidate_plan,
        construction_record,
        benchmark_record,
        validation_record,
    )
    staged = load_staged_candidate(
        candidate_workspace_root,
        candidate_plan.candidate_id,
        package_dir=package_dir,
    )
    if (
        staged.record != construction_record
        or staged.build_plan != candidate_plan.construction_requirements
    ):
        raise ModelCompileError(
            "promotion candidate construction evidence changed after validation"
        )
    integrity = validate_staged_candidate(
        staged.path,
        expected_candidate_id=candidate_plan.candidate_id,
        expected_build_plan=candidate_plan.construction_requirements,
    )

    loaded_benchmark = load_benchmark_evidence(
        benchmark_workspace_root,
        str(benchmark_record.to_json()["benchmark_id"]),
    )
    benchmark_plan, _benchmark_run, loaded_benchmark_record = (
        loaded_benchmark
    )
    if loaded_benchmark_record != benchmark_record:
        raise ModelCompileError(
            "promotion benchmark record does not match published raw evidence"
        )
    (
        validation_plan,
        prebenchmark_record,
        validation_benchmark_record,
        _validation_runs,
        loaded_validation_record,
    ) = load_validation_evidence(
        validation_workspace_root,
        str(validation_record.to_json()["validation_id"]),
    )
    if (
        loaded_validation_record != validation_record
        or validation_benchmark_record != benchmark_record
        or validation_plan.candidate_id != candidate_plan.candidate_id
        or validation_plan.to_json()["construction_record_digest"]
        != construction_record.digest
    ):
        raise ModelCompileError(
            "promotion validation evidence does not match candidate and benchmark"
        )
    (
        prebenchmark_plan,
        loaded_prebenchmark_record,
        _sanity_run,
    ) = load_prebenchmark_evidence(
        validation_workspace_root,
        str(prebenchmark_record.to_json()["prebenchmark_id"]),
    )
    if (
        loaded_prebenchmark_record != prebenchmark_record
        or prebenchmark_plan != validation_plan
    ):
        raise ModelCompileError(
            "promotion prebenchmark evidence does not match validation"
        )
    benchmark = benchmark_record.to_json()
    validation = validation_record.to_json()
    if (
        benchmark["decision"] != "materially_faster"
        or any(
            workload["decision"] != "materially_faster"
            for workload in benchmark["workloads"]
        )
        or validation["status"] != "passed"
    ):
        raise ModelCompileError(
            "only materially faster, fully validated candidates are promotable"
        )

    profiles = tuple(dict(profile) for profile in hardware_profiles)
    for profile in profiles:
        validate_contract(
            profile,
            expected_schema=HARDWARE_PROCESS_PROFILE_SCHEMA,
        )
    profiles = tuple(
        sorted(profiles, key=lambda profile: str(profile["profile_id"]))
    )
    if len({profile["profile_id"] for profile in profiles}) != len(
        profiles
    ):
        raise ModelCompileError(
            "promotion hardware profiles must be unique"
        )
    analysis_runs = _prepare_analysis_runs(
        analysis_run_directories,
        candidate=candidate_plan.candidate.to_json(),
        package_id=session.package_id,
    )
    runtime_predicate = _derive_runtime_predicate(
        benchmark_plan=benchmark_plan,
        hardware_profiles=profiles,
        candidate=candidate_plan.candidate.to_json(),
    )
    promoted_implementation_id = implementation_id(
        candidate_plan.candidate_id,
        runtime_predicate,
    )
    candidate = candidate_plan.candidate.to_json()
    graph = read_json(
        staged.path / "contracts" / "representation_graph.json"
    )
    target_lowering = read_json(
        staged.path / "contracts" / "target_lowering.json"
    )
    relowering = read_json(
        staged.path / "contracts" / "relowering_request.json"
    )
    artifact_integrity = integrity_evidence(integrity)
    comparison = {
        "exact_implementation_id": benchmark[
            "reference_implementation_id"
        ],
        "exact_contract_digest": benchmark_plan.implementation(
            "reference"
        )["contract_digest"],
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_decision": benchmark["decision"],
        "workloads": [
            {
                "workload_id": workload["workload_id"],
                "decision": workload["decision"],
                "paired": dict(workload["paired"]),
            }
            for workload in benchmark["workloads"]
        ],
        "validation_id": validation["validation_id"],
        "validation_status": validation["status"],
        "behavioral_contract": dict(validation["behavioral_contract"]),
    }
    provenance = {
        "provider": dict(candidate["provider"]),
        "descriptor_id": candidate["descriptor_id"],
        "evidence_refs": list(candidate["evidence_refs"]),
        "analysis_runs": [
            {
                "run_id": prepared.run.run_id,
                "run_digest": prepared.run_digest,
                "cited_evidence_ids": list(
                    prepared.cited_evidence_ids
                ),
            }
            for prepared in analysis_runs
        ],
        "hardware_profiles": [
            {
                "profile_id": profile["profile_id"],
                "profile_digest": contract_digest(profile),
            }
            for profile in profiles
        ],
        "representation_graph_digest": contract_digest(graph),
        "target_lowering_digest": contract_digest(target_lowering),
        "relowering_request_digest": contract_digest(relowering),
    }
    decision_document = {
        "schema": PROMOTION_DECISION_SCHEMA,
        "promotion_id": "",
        "candidate_id": candidate_plan.candidate_id,
        "implementation_id": promoted_implementation_id,
        "scope_ids": list(candidate["scope_ids"]),
        "source_contract_digests": list(
            candidate["source_contract_digests"]
        ),
        "candidate_contract_digest": candidate_plan.candidate.digest,
        "construction_record_digest": construction_record.digest,
        "prebenchmark_record_digest": prebenchmark_record.digest,
        "benchmark_record_digest": benchmark_record.digest,
        "validation_record_digest": validation_record.digest,
        "runtime_predicate": runtime_predicate.to_json(),
        "artifact_integrity": artifact_integrity,
        "comparison": comparison,
        "provenance": provenance,
        "decision": "promote",
        "reason": reason,
    }
    decision_document["promotion_id"] = promotion_decision_id(
        decision_document
    )
    decision = PromotionDecision.from_json(decision_document)
    root_ref = (
        "optimization/implementations/"
        f"{promoted_implementation_id}"
    )
    benchmark_ref = (
        f"{root_ref}/evidence/benchmarks/{benchmark['benchmark_id']}"
    )
    validation_ref = (
        f"{root_ref}/evidence/validations/{validation['validation_id']}"
    )
    registry_entry = {
        "implementation_id": promoted_implementation_id,
        "candidate_id": candidate_plan.candidate_id,
        "scope_ids": list(candidate["scope_ids"]),
        "source_contract_digests": list(
            candidate["source_contract_digests"]
        ),
        "representation": dict(candidate["representation"]),
        "behavioral_contract": dict(candidate["behavioral_contract"]),
        "runtime_predicate": runtime_predicate.to_json(),
        "artifact_bundle": {
            "root_ref": root_ref,
            "candidate_integrity_ref": f"{root_ref}/candidate/integrity.json",
            "mount_plan_ref": (
                f"{root_ref}/candidate/contracts/mount_plan.json"
            ),
            "candidate_integrity_digest": artifact_integrity["digest"],
            "artifact_count": artifact_integrity["file_count"],
        },
        "evidence": {
            "promotion_decision_ref": f"{root_ref}/promotion.json",
            "candidate_contract_ref": (
                f"{root_ref}/candidate/contracts/candidate.json"
            ),
            "construction_record_ref": (
                f"{root_ref}/construction_record.json"
            ),
            "prebenchmark_record_ref": (
                f"{root_ref}/evidence/prebenchmark/"
                f"{prebenchmark_record.to_json()['prebenchmark_id']}/"
                "record.json"
            ),
            "benchmark_record_ref": f"{benchmark_ref}/record.json",
            "validation_record_ref": f"{validation_ref}/record.json",
            "analysis_run_refs": [
                {
                    "run_id": prepared.run.run_id,
                    "artifact_ref": (
                        f"{root_ref}/evidence/analysis/"
                        f"{prepared.run.run_id}"
                    ),
                }
                for prepared in analysis_runs
            ],
            "hardware_profile_refs": [
                {
                    "profile_id": profile["profile_id"],
                    "artifact_ref": (
                        f"{root_ref}/evidence/hardware/"
                        f"{profile['profile_id']}.json"
                    ),
                }
                for profile in profiles
            ],
        },
        "provenance": provenance,
        "comparison": comparison,
        "decision_reason": reason,
    }
    from nerve.representation_optimizer.promotion.contracts import (
        validate_implementation_registry_entry,
    )

    validate_implementation_registry_entry(registry_entry)
    promoted_session = session.transition_candidate(
        candidate_plan.candidate_id,
        CandidateState.PROMOTABLE,
        evidence_refs=(
            f"promotions/{decision.promotion_id}.json",
        ),
        reason=reason,
    )
    return PreparedPromotion(
        candidate_plan=candidate_plan,
        construction_record=construction_record,
        prebenchmark_record=prebenchmark_record,
        benchmark_record=benchmark_record,
        validation_record=validation_record,
        runtime_predicate=runtime_predicate,
        decision=decision,
        registry_entry=registry_entry,
        staged_candidate=staged,
        benchmark_evidence_path=(
            benchmark_workspace_root.resolve()
            / "benchmarks"
            / benchmark["benchmark_id"]
        ),
        prebenchmark_evidence_path=(
            validation_workspace_root.resolve()
            / "prebenchmark"
            / prebenchmark_record.to_json()["prebenchmark_id"]
        ),
        validation_evidence_path=(
            validation_workspace_root.resolve()
            / "validations"
            / validation["validation_id"]
        ),
        analysis_runs=analysis_runs,
        hardware_profiles=profiles,
        session=promoted_session,
    )


def _prepare_analysis_runs(
    directories: Iterable[Path],
    *,
    candidate: Json,
    package_id: str,
) -> tuple[PreparedAnalysisRun, ...]:
    candidate_evidence = set(candidate["evidence_refs"])
    scope_sources = dict(
        zip(
            candidate["scope_ids"],
            candidate["source_contract_digests"],
            strict=True,
        )
    )
    prepared = []
    covered_evidence: set[str] = set()
    for directory in directories:
        path = directory.resolve()
        run = validate_analysis_run_directory(path)
        document = run.document
        scope_id = str(document["scope_id"])
        if (
            document["package_id"] != package_id
            or scope_sources.get(scope_id)
            != document["source_contract_digest"]
        ):
            raise ModelCompileError(
                "promotion analysis run does not belong to the candidate source"
            )
        evidence_by_id = {
            str(evidence["evidence_id"]): evidence
            for evidence in run.evidence
        }
        cited = tuple(
            sorted(candidate_evidence.intersection(evidence_by_id))
        )
        if not cited:
            raise ModelCompileError(
                "promotion received an analysis run not cited by the candidate"
            )
        if covered_evidence.intersection(cited):
            raise ModelCompileError(
                "promotion analysis evidence is supplied by multiple runs"
            )
        for evidence_id in cited:
            evidence = evidence_by_id[evidence_id]
            if (
                evidence["scope_id"] != scope_id
                or evidence["source_contract_digest"]
                != document["source_contract_digest"]
            ):
                raise ModelCompileError(
                    "promotion analysis evidence does not match its run"
                )
        covered_evidence.update(cited)
        prepared.append(
            PreparedAnalysisRun(
                path=path,
                run=run,
                run_digest=contract_digest(document),
                cited_evidence_ids=cited,
            )
        )
    if covered_evidence != candidate_evidence:
        raise ModelCompileError(
            "promotion does not include every analysis record cited by the candidate"
        )
    result = tuple(
        sorted(prepared, key=lambda item: item.run.run_id)
    )
    if len({item.run.run_id for item in result}) != len(result):
        raise ModelCompileError(
            "promotion contains duplicate analysis runs"
        )
    return result


def _derive_runtime_predicate(
    *,
    benchmark_plan,
    hardware_profiles: tuple[Json, ...],
    candidate: Json,
) -> RuntimeImplementationPredicate:
    if not hardware_profiles:
        raise ModelCompileError(
            "promotion requires the benchmarked hardware profiles"
        )
    for profile in hardware_profiles:
        validate_contract(
            profile,
            expected_schema=HARDWARE_PROCESS_PROFILE_SCHEMA,
        )
    expected_devices = sorted(
        (
            {
                "device_id": profile["hardware_identity"][
                    "stable_device_id"
                ],
                "hardware_profile_digest": contract_digest(profile),
                "capability_class": profile["capability_class"],
                "api": profile["provenance"]["api"],
            }
            for profile in hardware_profiles
        ),
        key=lambda device: device["device_id"],
    )
    matched = benchmark_plan.matched_conditions
    if matched["devices"] != expected_devices:
        raise ModelCompileError(
            "promotion hardware profiles do not match benchmark evidence"
        )
    workloads = [
        workload.to_json() for workload in benchmark_plan.workloads
    ]
    regimes = [workload["regime"] for workload in workloads]
    phases = sorted(
        {str(regime["execution_phase"]) for regime in regimes}
    )
    boundaries = {
        str(regime["boundary_mode"]) for regime in regimes
    }
    device_count = len(expected_devices)
    if boundaries == {"local"} and device_count == 1:
        placement_mode = "local"
    elif boundaries == {"cross_device"} or device_count > 1:
        placement_mode = "distributed"
    else:
        placement_mode = "either"
    target = candidate["target_predicate"]
    required_processes = _optional_string_list(
        target,
        "required_processes",
    )
    required_features = _optional_string_list(
        target,
        "required_features",
    )
    required_interconnects = _optional_string_list(
        target,
        "required_interconnects",
    )
    return create_runtime_implementation_predicate(
        capability_classes=(
            profile["capability_class"] for profile in hardware_profiles
        ),
        device_kinds=(
            profile["hardware_identity"]["device_kind"]
            for profile in hardware_profiles
        ),
        apis=(
            profile["provenance"]["api"]
            for profile in hardware_profiles
        ),
        required_processes=required_processes,
        required_features=required_features,
        execution_phases=phases,
        activation_batch_minimum=min(
            int(regime["activation_batch_width"])
            for regime in regimes
        ),
        activation_batch_maximum=max(
            int(regime["activation_batch_width"])
            for regime in regimes
        ),
        context_activations_minimum=min(
            int(regime["context_size"]) for regime in regimes
        ),
        context_activations_maximum=max(
            int(regime["context_size"]) for regime in regimes
        ),
        state_activations_minimum=min(
            int(regime["state_size"]) for regime in regimes
        ),
        state_activations_maximum=max(
            int(regime["state_size"]) for regime in regimes
        ),
        placement_mode=placement_mode,
        minimum_device_count=device_count,
        maximum_device_count=device_count,
        required_interconnects=required_interconnects,
    )


def _optional_string_list(document: Json, name: str) -> tuple[str, ...]:
    value = document.get(name, [])
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ModelCompileError(
            f"candidate target predicate {name!r} must be a string list"
        )
    return tuple(sorted(set(value)))


def _validate_record_headers(
    candidate_plan: ProviderCandidatePlan,
    construction_record: ContractDocument,
    benchmark_record: ContractDocument,
    validation_record: ContractDocument,
) -> None:
    candidate_id = candidate_plan.candidate_id
    construction = construction_record.to_json()
    benchmark = benchmark_record.to_json()
    validation = validation_record.to_json()
    if (
        construction_record.schema != CANDIDATE_CONSTRUCTION_SCHEMA
        or construction["candidate_id"] != candidate_id
        or construction["status"] != "completed"
        or benchmark_record.schema != BENCHMARK_RECORD_SCHEMA
        or benchmark["candidate_id"] != candidate_id
        or benchmark["construction_record_digest"]
        != construction_record.digest
        or validation_record.schema != VALIDATION_RECORD_SCHEMA
        or validation["candidate_id"] != candidate_id
        or validation["construction_record_digest"]
        != construction_record.digest
        or validation["benchmark_record_digest"]
        != benchmark_record.digest
    ):
        raise ModelCompileError(
            "candidate construction, benchmark, and validation records do not match"
        )


def _require_candidate_state(
    session: OptimizationSession,
    candidate_id: str,
    expected: CandidateState,
) -> None:
    matching = [
        candidate
        for candidate in session.candidates
        if candidate.candidate_id == candidate_id
    ]
    if len(matching) != 1 or matching[0].state != expected:
        raise ModelCompileError(
            "candidate must be behaviorally validated before promotion"
        )
