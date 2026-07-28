from __future__ import annotations

from typing import Iterable, Mapping

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.benchmarking.contracts import BenchmarkPlan
from nerve.representation_optimizer.contracts import (
    CANDIDATE_CONSTRUCTION_SCHEMA,
    ContractDocument,
    contract_digest,
)
from nerve.representation_optimizer.providers.types import ProviderCandidatePlan
from nerve.representation_optimizer.validation.contracts import (
    BEHAVIORAL_ERROR_CONTRACT_SCHEMA,
    VALIDATION_COVERAGE_KINDS,
    VALIDATION_PLAN_SCHEMA,
    VALIDATION_REQUIREMENTS_SCHEMA,
    BehavioralErrorContract,
    ValidationPlan,
    ValidationRequirements,
    behavioral_error_contract_id,
    validation_check_id,
    validation_plan_id,
    validation_requirements_id,
)


def create_behavioral_error_contract(
    *,
    validity_predicates: Json,
    metric_limits: Mapping[str, tuple[float, str, Iterable[str]]],
    correction_mode: str,
    correction_trigger_metrics: Iterable[str],
    correction_action: str,
) -> BehavioralErrorContract:
    metrics = [
        {
            "name": name,
            "maximum_error": maximum_error,
            "unit": unit,
            "coverage": sorted(set(coverage)),
        }
        for name, (maximum_error, unit, coverage) in sorted(
            metric_limits.items()
        )
    ]
    document = {
        "schema": BEHAVIORAL_ERROR_CONTRACT_SCHEMA,
        "contract_id": "",
        "validity_predicates": dict(validity_predicates),
        "metrics": metrics,
        "correction_policy": {
            "mode": correction_mode,
            "trigger_metrics": sorted(set(correction_trigger_metrics)),
            "action": correction_action,
        },
    }
    document["contract_id"] = behavioral_error_contract_id(document)
    return BehavioralErrorContract.from_json(document)


def create_validation_check(
    *,
    name: str,
    stage: str,
    kind: str,
    coverage: Iterable[str],
    execution_scope: str,
    activation_batch_width: int,
    context_size: int,
    context_size_basis: Json,
    state_size: int,
    boundary_mode: str,
    input_artifact: Json,
    initial_state_artifact: Json | None,
    controls: Json,
    seeds: Iterable[int],
    step_unit: str,
    completion_condition: str,
    minimum_steps: int | None,
    output_allowance: int | None,
    output_allowance_basis: Json,
    metrics: Iterable[str],
) -> Json:
    document = {
        "check_id": "",
        "name": name,
        "stage": stage,
        "kind": kind,
        "coverage": sorted(set(coverage)),
        "regime": {
            "execution_scope": execution_scope,
            "activation_batch_width": activation_batch_width,
            "context_size": context_size,
            "context_size_basis": dict(context_size_basis),
            "state_size": state_size,
            "boundary_mode": boundary_mode,
        },
        "input": dict(input_artifact),
        "initial_state": (
            None
            if initial_state_artifact is None
            else dict(initial_state_artifact)
        ),
        "controls": dict(controls),
        "seeds": sorted(set(seeds)),
        "horizon": {
            "unit": step_unit,
            "completion_condition": completion_condition,
            "minimum_steps": minimum_steps,
            "output_allowance": output_allowance,
            "output_allowance_basis": dict(output_allowance_basis),
        },
        "metrics": sorted(set(metrics)),
    }
    document["check_id"] = validation_check_id(document)
    return document


def create_validation_requirements(
    *,
    candidate_id: str,
    source_contract_digests: Iterable[str],
    proof_verifiers: Mapping[str, str],
    checks: Iterable[Json],
    not_applicable_reasons: Mapping[str, str],
    counterexamples: Iterable[Json] = (),
) -> ValidationRequirements:
    ordered_checks = sorted(
        (dict(check) for check in checks),
        key=lambda check: check["check_id"],
    )
    check_ids_by_coverage: dict[str, list[str]] = {
        kind: [] for kind in VALIDATION_COVERAGE_KINDS
    }
    for check in ordered_checks:
        for kind in check["coverage"]:
            if kind in check_ids_by_coverage:
                check_ids_by_coverage[kind].append(check["check_id"])
    coverage = []
    for kind in VALIDATION_COVERAGE_KINDS:
        check_ids = sorted(set(check_ids_by_coverage[kind]))
        if check_ids:
            if kind in not_applicable_reasons:
                raise ModelCompileError(
                    f"validation coverage {kind!r} cannot be both required "
                    "and not applicable"
                )
            coverage.append(
                {
                    "kind": kind,
                    "applicability": "required",
                    "check_ids": check_ids,
                    "reason": None,
                }
            )
        else:
            reason = not_applicable_reasons.get(kind)
            if not reason:
                raise ModelCompileError(
                    f"validation coverage {kind!r} needs a check or an "
                    "explicit not-applicable reason"
                )
            coverage.append(
                {
                    "kind": kind,
                    "applicability": "not_applicable",
                    "check_ids": [],
                    "reason": reason,
                }
            )
    document = {
        "schema": VALIDATION_REQUIREMENTS_SCHEMA,
        "requirements_id": "",
        "candidate_id": candidate_id,
        "source_contract_digests": sorted(set(source_contract_digests)),
        "proofs": [
            {"obligation": obligation, "verifier_id": verifier_id}
            for obligation, verifier_id in sorted(proof_verifiers.items())
        ],
        "checks": ordered_checks,
        "coverage": coverage,
        "counterexamples": sorted(
            (dict(reference) for reference in counterexamples),
            key=lambda reference: reference["path"],
        ),
    }
    document["requirements_id"] = validation_requirements_id(document)
    return ValidationRequirements.from_json(document)


def build_validation_plan(
    *,
    candidate_plan: ProviderCandidatePlan,
    construction_record: ContractDocument,
    benchmark_plan: BenchmarkPlan,
) -> ValidationPlan:
    construction = construction_record.to_json()
    candidate = candidate_plan.candidate.to_json()
    benchmark = benchmark_plan.to_json()
    if (
        construction["schema"] != CANDIDATE_CONSTRUCTION_SCHEMA
        or construction["status"] != "completed"
        or construction["candidate_id"] != candidate_plan.candidate_id
        or benchmark["candidate_id"] != candidate_plan.candidate_id
        or benchmark["construction_record_digest"]
        != construction_record.digest
    ):
        raise ModelCompileError(
            "validation planning requires matching completed construction "
            "and benchmark plans"
        )
    requirements = candidate_plan.validation_requirements
    if (
        requirements.candidate_id != candidate_plan.candidate_id
        or requirements.to_json()["source_contract_digests"]
        != sorted(candidate["source_contract_digests"])
    ):
        raise ModelCompileError(
            "validation requirements do not match their representation candidate"
        )
    requirements_document = requirements.to_json()
    document = {
        "schema": VALIDATION_PLAN_SCHEMA,
        "plan_id": "",
        "candidate_id": candidate_plan.candidate_id,
        "source_contract_digests": sorted(
            candidate["source_contract_digests"]
        ),
        "construction_record_digest": construction_record.digest,
        "requirements_digest": contract_digest(requirements_document),
        "behavioral_contract": dict(candidate["behavioral_contract"]),
        "implementations": {
            role: dict(benchmark["implementations"][role])
            for role in ("reference", "candidate")
        },
        "matched_conditions": dict(benchmark["matched_conditions"]),
        "matched_conditions_digest": benchmark[
            "matched_conditions_digest"
        ],
        "proofs": list(requirements_document["proofs"]),
        "checks": list(requirements_document["checks"]),
        "coverage": list(requirements_document["coverage"]),
        "counterexamples": list(requirements_document["counterexamples"]),
    }
    document["plan_id"] = validation_plan_id(document)
    return ValidationPlan.from_json(document)
