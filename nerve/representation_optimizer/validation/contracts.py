from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from nerve.compilation import Json
from nerve.representation_optimizer.benchmarking.contracts import (
    validate_matched_conditions,
)
from nerve.representation_optimizer.contracts import (
    ContractValidationError,
    DEVICE_STATE_DIGEST_SCHEMA,
    canonical_json_bytes,
    contract_digest,
    stable_contract_id,
)


BEHAVIORAL_ERROR_CONTRACT_SCHEMA = (
    "nerve.optimizer.behavioral_error_contract.v1"
)
VALIDATION_REQUIREMENTS_SCHEMA = "nerve.optimizer.validation_requirements.v2"
VALIDATION_PLAN_SCHEMA = "nerve.optimizer.validation_plan.v4"
PROOF_RESULT_SCHEMA = "nerve.optimizer.proof_result.v1"
VALIDATION_ROLE_RESULT_SCHEMA = "nerve.optimizer.validation_role_result.v2"
VALIDATION_OBSERVATION_SCHEMA = "nerve.optimizer.validation_observation.v3"
VALIDATION_RESIDENCY_EVENT_SCHEMA = (
    "nerve.optimizer.validation_residency_event.v3"
)
VALIDATION_RUN_SCHEMA = "nerve.optimizer.validation_run.v4"
PREBENCHMARK_RECORD_SCHEMA = "nerve.optimizer.prebenchmark_record.v1"
VALIDATION_EVIDENCE_INTEGRITY_SCHEMA = (
    "nerve.optimizer.validation_evidence_integrity.v1"
)

VALIDATION_STAGES = ("sanity", "full_local", "whole_model")
VALIDATION_CHECK_KINDS = (
    "component_comparison",
    "state_transition",
    "teacher_forced",
    "free_running",
    "reasoning_conversation",
    "lifecycle_operation",
    "graph_edit",
    "placement",
    "counterexample",
)
VALIDATION_COVERAGE_KINDS = (
    "component_output_error",
    "state_transition_consistency",
    "distribution_divergence",
    "top_k_overlap",
    "rank_stability",
    "route_recall",
    "memory_recall",
    "candidate_recall",
    "confidence_calibration",
    "correction_calibration",
    "teacher_forced_sequences",
    "free_running_long_horizon",
    "multiple_fixed_seeds",
    "reasoning_enabled_conversations",
    "long_context",
    "long_output",
    "interruption",
    "snapshot",
    "fork",
    "rollback",
    "resumption",
    "graph_edits",
    "alternative_placements",
    "adversarial_counterexamples",
)
VALIDATION_FUNNEL_STAGE_NAMES = (
    "static_contracts_and_artifacts",
    "exact_algebraic_proof",
    "cheap_numerical_and_state_sanity",
    "matched_performance",
    "full_local_behavior",
    "whole_model_free_running",
    "whole_model_product_performance",
)

_ARTIFACT_DIGEST_PREFIX = "nerve.optimizer.artifact_sha256.v1:"
_CONTRACT_DIGEST_PREFIX = "nerve.optimizer.canonical_json_sha256.v1:"
_IMPLEMENTATION_ROLES = ("reference", "candidate")
_PROOF_STATUSES = ("proven", "disproven", "inconclusive")
_RUN_STATUSES = ("completed", "failed", "cancelled")
_STAGE_STATUSES = ("passed", "failed", "not_applicable", "not_run")
_HORIZON_COMPLETION_CONDITIONS = (
    "minimum_steps",
    "all_fixture_turns",
    "semantic_stop_or_allowance_per_turn",
)
_SEMANTIC_STOP_REASONS = ("eos", "output_allowance")
_FIXTURE_STOP_REASONS = ("fixture_completed",)
_COVERAGE_CHECK_CONSTRAINTS = {
    "component_output_error": (
        {"sanity", "full_local"},
        {"component_comparison"},
    ),
    "state_transition_consistency": (
        {"sanity", "full_local"},
        {"component_comparison", "state_transition"},
    ),
    "distribution_divergence": (
        {"sanity", "full_local", "whole_model"},
        {"component_comparison", "teacher_forced", "free_running"},
    ),
    "top_k_overlap": (
        {"sanity", "full_local", "whole_model"},
        {"component_comparison", "teacher_forced", "free_running"},
    ),
    "rank_stability": (
        {"sanity", "full_local", "whole_model"},
        {"component_comparison", "teacher_forced", "free_running"},
    ),
    "route_recall": (
        {"sanity", "full_local"},
        {"component_comparison", "teacher_forced"},
    ),
    "memory_recall": (
        {"sanity", "full_local", "whole_model"},
        {
            "component_comparison",
            "state_transition",
            "teacher_forced",
            "free_running",
        },
    ),
    "candidate_recall": (
        {"sanity", "full_local"},
        {"component_comparison", "teacher_forced"},
    ),
    "confidence_calibration": (
        {"full_local", "whole_model"},
        {"teacher_forced", "free_running", "reasoning_conversation"},
    ),
    "correction_calibration": (
        {"full_local", "whole_model"},
        {"teacher_forced", "free_running", "reasoning_conversation"},
    ),
    "teacher_forced_sequences": (
        {"full_local"},
        {"teacher_forced"},
    ),
    "free_running_long_horizon": (
        {"whole_model"},
        {"free_running", "reasoning_conversation"},
    ),
    "multiple_fixed_seeds": (
        {"full_local", "whole_model"},
        set(VALIDATION_CHECK_KINDS),
    ),
    "reasoning_enabled_conversations": (
        {"whole_model"},
        {"reasoning_conversation"},
    ),
    "long_context": (
        {"whole_model"},
        {"free_running", "reasoning_conversation"},
    ),
    "long_output": (
        {"whole_model"},
        {"free_running", "reasoning_conversation"},
    ),
    "interruption": (
        {"full_local"},
        {"lifecycle_operation"},
    ),
    "snapshot": (
        {"full_local"},
        {"lifecycle_operation"},
    ),
    "fork": ({"full_local"}, {"lifecycle_operation"}),
    "rollback": ({"full_local"}, {"lifecycle_operation"}),
    "resumption": ({"full_local"}, {"lifecycle_operation"}),
    "graph_edits": ({"full_local"}, {"graph_edit"}),
    "alternative_placements": (
        {"full_local", "whole_model"},
        {"placement"},
    ),
    "adversarial_counterexamples": (
        {"full_local", "whole_model"},
        {"counterexample"},
    ),
}
_UNIVERSAL_VALIDATION_COVERAGE = frozenset(
    {
        "component_output_error",
        "teacher_forced_sequences",
        "free_running_long_horizon",
        "multiple_fixed_seeds",
        "long_context",
        "long_output",
        "graph_edits",
        "alternative_placements",
    }
)


class ValidationContractError(ContractValidationError):
    """A proof or behavioral-validation contract is invalid."""


@dataclass(frozen=True)
class BehavioralErrorContract:
    _document: Json

    @classmethod
    def from_json(cls, document: Json) -> BehavioralErrorContract:
        normalized = deepcopy(document)
        validate_behavioral_error_contract(normalized)
        return cls(normalized)

    @property
    def metric_limits(self) -> dict[str, float]:
        return {
            str(metric["name"]): float(metric["maximum_error"])
            for metric in self._document["metrics"]
        }

    def to_json(self) -> Json:
        return deepcopy(self._document)


@dataclass(frozen=True)
class ValidationRequirements:
    _document: Json

    @classmethod
    def from_json(cls, document: Json) -> ValidationRequirements:
        normalized = deepcopy(document)
        validate_validation_requirements(normalized)
        return cls(normalized)

    @property
    def requirements_id(self) -> str:
        return str(self._document["requirements_id"])

    @property
    def candidate_id(self) -> str:
        return str(self._document["candidate_id"])

    @property
    def proofs(self) -> tuple[Json, ...]:
        return tuple(deepcopy(self._document["proofs"]))

    @property
    def checks(self) -> tuple[Json, ...]:
        return tuple(deepcopy(self._document["checks"]))

    def checks_for_stage(self, stage: str) -> tuple[Json, ...]:
        if stage not in VALIDATION_STAGES:
            raise ValidationContractError(
                f"unsupported validation stage {stage!r}"
            )
        return tuple(
            deepcopy(check)
            for check in self._document["checks"]
            if check["stage"] == stage
        )

    def to_json(self) -> Json:
        return deepcopy(self._document)


@dataclass(frozen=True)
class ValidationPlan:
    _document: Json

    @classmethod
    def from_json(cls, document: Json) -> ValidationPlan:
        normalized = deepcopy(document)
        validate_validation_plan(normalized)
        return cls(normalized)

    @property
    def plan_id(self) -> str:
        return str(self._document["plan_id"])

    @property
    def candidate_id(self) -> str:
        return str(self._document["candidate_id"])

    @property
    def behavioral_contract(self) -> Json:
        return deepcopy(self._document["behavioral_contract"])

    @property
    def proofs(self) -> tuple[Json, ...]:
        return tuple(deepcopy(self._document["proofs"]))

    def checks_for_stage(self, stage: str) -> tuple[Json, ...]:
        if stage not in VALIDATION_STAGES:
            raise ValidationContractError(
                f"unsupported validation stage {stage!r}"
            )
        return tuple(
            deepcopy(check)
            for check in self._document["checks"]
            if check["stage"] == stage
        )

    def implementation(self, role: str) -> Json:
        if role not in _IMPLEMENTATION_ROLES:
            raise ValidationContractError(
                f"unsupported validation implementation role {role!r}"
            )
        return deepcopy(self._document["implementations"][role])

    def to_json(self) -> Json:
        return deepcopy(self._document)


@dataclass(frozen=True)
class ProofResult:
    _document: Json

    @classmethod
    def from_json(cls, document: Json) -> ProofResult:
        normalized = deepcopy(document)
        validate_proof_result(normalized)
        return cls(normalized)

    @property
    def status(self) -> str:
        return str(self._document["status"])

    def to_json(self) -> Json:
        return deepcopy(self._document)


@dataclass(frozen=True)
class ValidationObservation:
    _document: Json

    @classmethod
    def from_json(cls, document: Json) -> ValidationObservation:
        normalized = deepcopy(document)
        validate_validation_observation(normalized)
        return cls(normalized)

    @property
    def observation_id(self) -> str:
        return str(self._document["observation_id"])

    def to_json(self) -> Json:
        return deepcopy(self._document)


@dataclass(frozen=True)
class ValidationRoleResult:
    _document: Json

    @classmethod
    def from_json(cls, document: Json) -> ValidationRoleResult:
        normalized = deepcopy(document)
        validate_validation_role_result(normalized)
        return cls(normalized)

    def to_json(self) -> Json:
        return deepcopy(self._document)


@dataclass(frozen=True)
class ValidationResidencyEvent:
    _document: Json

    @classmethod
    def from_json(cls, document: Json) -> ValidationResidencyEvent:
        normalized = deepcopy(document)
        validate_validation_residency_event(normalized)
        return cls(normalized)

    def to_json(self) -> Json:
        return deepcopy(self._document)


@dataclass(frozen=True)
class ValidationRun:
    _document: Json

    @classmethod
    def from_json(cls, document: Json) -> ValidationRun:
        normalized = deepcopy(document)
        validate_validation_run(normalized)
        return cls(normalized)

    @property
    def status(self) -> str:
        return str(self._document["status"])

    def to_json(self) -> Json:
        return deepcopy(self._document)


def behavioral_error_contract_id(document: Json) -> str:
    return _content_id(
        "behavioral_error_contract",
        "contract_id",
        document,
    )


def validation_requirements_id(document: Json) -> str:
    return _content_id(
        "validation_requirements",
        "requirements_id",
        document,
    )


def validation_check_id(document: Json) -> str:
    return _content_id("validation_check", "check_id", document)


def validation_plan_id(document: Json) -> str:
    return _content_id("validation_plan", "plan_id", document)


def proof_result_id(document: Json) -> str:
    return _content_id("proof_result", "proof_id", document)


def validation_observation_id(document: Json) -> str:
    return _content_id(
        "validation_observation",
        "observation_id",
        document,
    )


def validation_role_result_id(document: Json) -> str:
    return _content_id(
        "validation_role_result",
        "result_id",
        document,
    )


def validation_residency_event_id(document: Json) -> str:
    return _content_id(
        "validation_residency",
        "event_id",
        document,
    )


def validation_run_id(document: Json) -> str:
    return _content_id("validation_run", "run_id", document)


def prebenchmark_record_id(document: Json) -> str:
    return _content_id(
        "prebenchmark_validation",
        "prebenchmark_id",
        document,
    )


def validation_record_id(document: Json) -> str:
    return _content_id("validation", "validation_id", document)


def validate_behavioral_error_contract(document: Json) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {
            "schema",
            "contract_id",
            "validity_predicates",
            "metrics",
            "correction_policy",
        },
        "behavioral error contract",
    )
    _schema(
        document,
        BEHAVIORAL_ERROR_CONTRACT_SCHEMA,
        "behavioral error contract",
    )
    _stable_id(
        document["contract_id"],
        "behavioral_error_contract",
        "contract_id",
    )
    if not _object(
        document["validity_predicates"],
        "validity_predicates",
    ):
        raise ValidationContractError(
            "behavioral error contract validity predicates must not be empty"
        )
    metrics = _list(document["metrics"], "metrics")
    if not metrics:
        raise ValidationContractError(
            "behavioral error contract must declare at least one metric"
        )
    names: list[str] = []
    covered: set[str] = set()
    for index, raw_metric in enumerate(metrics):
        path = f"metrics[{index}]"
        metric = _object(raw_metric, path)
        _fields(
            metric,
            {"name", "maximum_error", "unit", "coverage"},
            path,
        )
        names.append(_text(metric["name"], f"{path}.name"))
        _finite_nonnegative(
            metric["maximum_error"],
            f"{path}.maximum_error",
        )
        _text(metric["unit"], f"{path}.unit")
        coverage = _sorted_unique_strings(
            metric["coverage"],
            f"{path}.coverage",
            nonempty=True,
        )
        unsupported = sorted(set(coverage) - set(VALIDATION_COVERAGE_KINDS))
        if unsupported:
            raise ValidationContractError(
                f"{path}.coverage contains unsupported values {unsupported}"
            )
        covered.update(coverage)
    if names != sorted(set(names)):
        raise ValidationContractError(
            "behavioral error contract metrics must be sorted and unique"
        )
    correction = _object(
        document["correction_policy"],
        "correction_policy",
    )
    _fields(
        correction,
        {"mode", "trigger_metrics", "action"},
        "correction_policy",
    )
    if correction["mode"] not in {"reject", "fallback_exact", "correct"}:
        raise ValidationContractError(
            "correction_policy.mode is unsupported"
        )
    triggers = _sorted_unique_strings(
        correction["trigger_metrics"],
        "correction_policy.trigger_metrics",
        nonempty=True,
    )
    if not set(triggers) <= set(names):
        raise ValidationContractError(
            "correction policy references undeclared error metrics"
        )
    _text(correction["action"], "correction_policy.action")
    expected = behavioral_error_contract_id(document)
    if document["contract_id"] != expected:
        raise ValidationContractError(
            f"behavioral error contract id must be {expected!r}"
        )


def validate_validation_requirements(document: Json) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {
            "schema",
            "requirements_id",
            "candidate_id",
            "source_contract_digests",
            "proofs",
            "checks",
            "coverage",
            "counterexamples",
        },
        "validation requirements",
    )
    _schema(
        document,
        VALIDATION_REQUIREMENTS_SCHEMA,
        "validation requirements",
    )
    _stable_id(
        document["requirements_id"],
        "validation_requirements",
        "requirements_id",
    )
    _stable_id(document["candidate_id"], "candidate", "candidate_id")
    _digests(
        document["source_contract_digests"],
        "source_contract_digests",
        nonempty=True,
    )
    proofs = _list(document["proofs"], "proofs")
    obligations: list[str] = []
    for index, raw_proof in enumerate(proofs):
        path = f"proofs[{index}]"
        proof = _object(raw_proof, path)
        _fields(proof, {"obligation", "verifier_id"}, path)
        obligations.append(
            _text(proof["obligation"], f"{path}.obligation")
        )
        _text(proof["verifier_id"], f"{path}.verifier_id")
    if obligations != sorted(set(obligations)):
        raise ValidationContractError(
            "validation proof obligations must be sorted and unique"
        )

    checks = _list(document["checks"], "checks")
    if not checks:
        raise ValidationContractError(
            "validation requirements must declare checks"
        )
    check_by_id: dict[str, Json] = {}
    for index, raw_check in enumerate(checks):
        check = _object(raw_check, f"checks[{index}]")
        validate_validation_check(check)
        check_id = str(check["check_id"])
        if check_id in check_by_id:
            raise ValidationContractError(
                "validation requirements contain duplicate checks"
            )
        check_by_id[check_id] = check
    if list(check_by_id) != sorted(check_by_id):
        raise ValidationContractError(
            "validation checks must be sorted by check_id"
        )
    for stage in VALIDATION_STAGES:
        stage_checks = [
            check for check in checks if check["stage"] == stage
        ]
        if not stage_checks:
            raise ValidationContractError(
                f"validation requirements need at least one {stage!r} check"
            )
    product_checks = [
        check for check in checks if check["product_performance"]
    ]
    if len(product_checks) != 1:
        raise ValidationContractError(
            "validation requirements need exactly one product-performance "
            "check"
        )
    coverage = _list(document["coverage"], "coverage")
    coverage_names: list[str] = []
    coverage_by_kind: dict[str, Json] = {}
    for index, raw_coverage in enumerate(coverage):
        path = f"coverage[{index}]"
        entry = _object(raw_coverage, path)
        _fields(
            entry,
            {"kind", "applicability", "check_ids", "reason"},
            path,
        )
        kind = _text(entry["kind"], f"{path}.kind")
        coverage_names.append(kind)
        coverage_by_kind[kind] = entry
        if kind not in VALIDATION_COVERAGE_KINDS:
            raise ValidationContractError(
                f"{path}.kind is unsupported"
            )
        if entry["applicability"] not in {"required", "not_applicable"}:
            raise ValidationContractError(
                f"{path}.applicability is unsupported"
            )
        check_ids = _sorted_unique_strings(
            entry["check_ids"],
            f"{path}.check_ids",
        )
        unknown = sorted(set(check_ids) - set(check_by_id))
        if unknown:
            raise ValidationContractError(
                f"{path} references unknown validation checks {unknown}"
            )
        if entry["applicability"] == "required":
            if not check_ids or entry["reason"] is not None:
                raise ValidationContractError(
                    f"{path} required coverage needs checks and no reason"
                )
            for check_id in check_ids:
                if kind not in check_by_id[check_id]["coverage"]:
                    raise ValidationContractError(
                        f"{path} is not declared by check {check_id!r}"
                    )
                allowed_stages, allowed_kinds = (
                    _COVERAGE_CHECK_CONSTRAINTS[kind]
                )
                check = check_by_id[check_id]
                if (
                    check["stage"] not in allowed_stages
                    or check["kind"] not in allowed_kinds
                ):
                    raise ValidationContractError(
                        f"{path} is assigned to an incompatible "
                        f"{check['stage']!r}/{check['kind']!r} check"
                    )
                if kind in {
                    "teacher_forced_sequences",
                    "free_running_long_horizon",
                    "reasoning_enabled_conversations",
                    "long_context",
                    "long_output",
                    "interruption",
                    "snapshot",
                    "fork",
                    "rollback",
                    "resumption",
                    "graph_edits",
                    "alternative_placements",
                } and check["regime"]["execution_scope"] != "whole_model":
                    raise ValidationContractError(
                        f"{path} requires whole-model execution"
                    )
                if (
                    kind == "long_context"
                    and check["regime"]["context_size"] == 0
                ):
                    raise ValidationContractError(
                        f"{path} requires a non-zero declared context limit"
                    )
                if (
                    kind == "long_output"
                    and check["horizon"]["output_allowance"] is None
                ):
                    raise ValidationContractError(
                        f"{path} requires a declared output allowance"
                    )
                if (
                    kind == "multiple_fixed_seeds"
                    and len(check["seeds"]) < 2
                ):
                    raise ValidationContractError(
                        f"{path} requires at least two fixed seeds"
                    )
        else:
            if check_ids or not isinstance(entry["reason"], str) or not entry["reason"]:
                raise ValidationContractError(
                    f"{path} not-applicable coverage needs a reason and no checks"
                )
    if coverage_names != list(VALIDATION_COVERAGE_KINDS):
        raise ValidationContractError(
            "validation coverage must declare every coverage kind in canonical order"
        )
    missing_universal = sorted(
        kind
        for kind in _UNIVERSAL_VALIDATION_COVERAGE
        if coverage_by_kind[kind]["applicability"] != "required"
    )
    if missing_universal:
        raise ValidationContractError(
            "validation requirements cannot waive whole-pipeline coverage "
            f"{missing_universal}"
        )

    counterexamples = _artifact_refs(
        document["counterexamples"],
        "counterexamples",
    )
    if counterexamples:
        adversarial = next(
            entry
            for entry in coverage
            if entry["kind"] == "adversarial_counterexamples"
        )
        if adversarial["applicability"] != "required":
            raise ValidationContractError(
                "declared counterexamples require adversarial validation coverage"
            )
        counterexample_inputs = {
            check["input"]["path"]
            for check in checks
            if check["kind"] == "counterexample"
        }
        counterexample_paths = {
            reference["path"] for reference in counterexamples
        }
        if counterexample_inputs != counterexample_paths:
            raise ValidationContractError(
                "every counterexample artifact must have exactly one "
                "counterexample validation input"
            )
    expected = validation_requirements_id(document)
    if document["requirements_id"] != expected:
        raise ValidationContractError(
            f"validation requirements id must be {expected!r}"
        )


def validate_validation_check(document: Json) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {
            "check_id",
            "name",
            "stage",
            "kind",
            "product_performance",
            "coverage",
            "regime",
            "input",
            "initial_state",
            "controls",
            "seeds",
            "horizon",
            "comparison",
            "metrics",
        },
        "validation check",
    )
    _stable_id(document["check_id"], "validation_check", "check_id")
    _text(document["name"], "name")
    if document["stage"] not in VALIDATION_STAGES:
        raise ValidationContractError("validation check stage is unsupported")
    if document["kind"] not in VALIDATION_CHECK_KINDS:
        raise ValidationContractError("validation check kind is unsupported")
    product_performance = document["product_performance"]
    if not isinstance(product_performance, bool):
        raise ValidationContractError(
            "validation check product_performance must be boolean"
        )
    coverage = _sorted_unique_strings(
        document["coverage"],
        "coverage",
        nonempty=True,
    )
    unsupported = sorted(set(coverage) - set(VALIDATION_COVERAGE_KINDS))
    if unsupported:
        raise ValidationContractError(
            f"validation check coverage is unsupported: {unsupported}"
        )
    regime = _object(document["regime"], "regime")
    _fields(
        regime,
        {
            "execution_scope",
            "activation_batch_width",
            "context_size",
            "context_size_basis",
            "state_size",
            "boundary_mode",
        },
        "regime",
    )
    if regime["execution_scope"] not in {"component", "whole_model"}:
        raise ValidationContractError(
            "regime.execution_scope is unsupported"
        )
    _positive_integer(
        regime["activation_batch_width"],
        "regime.activation_batch_width",
    )
    context_size = _nonnegative_integer(
        regime["context_size"],
        "regime.context_size",
    )
    _limit_basis(
        regime["context_size_basis"],
        context_size,
        "regime.context_size_basis",
        zero_kind="not_applicable",
    )
    _nonnegative_integer(regime["state_size"], "regime.state_size")
    if regime["boundary_mode"] not in {"local", "cross_device"}:
        raise ValidationContractError(
            "regime.boundary_mode is unsupported"
        )
    _artifact_ref(document["input"], "input")
    if document["initial_state"] is not None:
        _artifact_ref(document["initial_state"], "initial_state")
    controls = _object(document["controls"], "controls")
    sampler = controls.get("sampler")
    if sampler is not None:
        sampler = _object(sampler, "controls.sampler")
        allowed_sampler_fields = {
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "presence_penalty",
            "repetition_penalty",
        }
        unsupported_sampler_fields = sorted(
            set(sampler) - allowed_sampler_fields
        )
        if not sampler or unsupported_sampler_fields:
            raise ValidationContractError(
                "controls.sampler is empty or has unsupported fields"
            )
        if "temperature" in sampler:
            _finite_positive(
                sampler["temperature"],
                "controls.sampler.temperature",
            )
        if "top_k" in sampler:
            _positive_integer(
                sampler["top_k"],
                "controls.sampler.top_k",
            )
        if "top_p" in sampler:
            top_p = _finite_positive(
                sampler["top_p"],
                "controls.sampler.top_p",
            )
            if top_p > 1:
                raise ValidationContractError(
                    "controls.sampler.top_p must not exceed one"
                )
        if "min_p" in sampler:
            min_p = _finite_nonnegative(
                sampler["min_p"],
                "controls.sampler.min_p",
            )
            if min_p > 1:
                raise ValidationContractError(
                    "controls.sampler.min_p must not exceed one"
                )
        if "presence_penalty" in sampler:
            _finite_number(
                sampler["presence_penalty"],
                "controls.sampler.presence_penalty",
            )
        if "repetition_penalty" in sampler:
            _finite_positive(
                sampler["repetition_penalty"],
                "controls.sampler.repetition_penalty",
            )
    seeds = _list(document["seeds"], "seeds")
    if (
        not seeds
        or seeds != sorted(set(seeds))
        or any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or seed > 0xFFFF_FFFF
            for seed in seeds
        )
    ):
        raise ValidationContractError(
            "validation check seeds must be sorted unique U32 values"
        )
    horizon = _object(document["horizon"], "horizon")
    _fields(
        horizon,
        {
            "unit",
            "completion_condition",
            "minimum_steps",
            "output_allowance",
            "output_allowance_basis",
        },
        "horizon",
    )
    _text(horizon["unit"], "horizon.unit")
    completion_condition = horizon["completion_condition"]
    if completion_condition not in _HORIZON_COMPLETION_CONDITIONS:
        raise ValidationContractError(
            "horizon completion condition is unsupported"
        )
    minimum_steps = horizon["minimum_steps"]
    if completion_condition == "minimum_steps":
        minimum_steps = _positive_integer(
            minimum_steps,
            "horizon.minimum_steps",
        )
    elif minimum_steps is not None:
        raise ValidationContractError(
            "turn-completion horizon cannot declare minimum steps"
        )
    if completion_condition == "all_fixture_turns" and (
        document["kind"]
        not in {
            "teacher_forced",
            "lifecycle_operation",
            "graph_edit",
            "placement",
        }
        or regime["execution_scope"] != "whole_model"
        or document["controls"].get("execution_mode")
        not in {"teacher_forced", "lifecycle_teacher_forced"}
    ):
        raise ValidationContractError(
            "fixture-turn horizon completion is incompatible with anything "
            "except teacher-forced whole-model execution"
        )
    if completion_condition == "semantic_stop_or_allowance_per_turn" and (
        document["kind"] not in {"free_running", "reasoning_conversation"}
        or regime["execution_scope"] != "whole_model"
        or document["controls"].get("execution_mode") != "conversation"
    ):
        raise ValidationContractError(
            "semantic horizon completion is incompatible with anything "
            "except free-running whole-model conversation execution"
        )
    allowance = horizon["output_allowance"]
    if allowance is None:
        basis = _object(
            horizon["output_allowance_basis"],
            "horizon.output_allowance_basis",
        )
        _fields(
            basis,
            {"kind"},
            "horizon.output_allowance_basis",
        )
        if basis["kind"] != "unlimited":
            raise ValidationContractError(
                "unbounded validation output requires an unlimited basis"
            )
    else:
        allowance = _positive_integer(
            allowance,
            "horizon.output_allowance",
        )
        if minimum_steps is not None and allowance < minimum_steps:
            raise ValidationContractError(
                "validation output allowance is below its minimum horizon"
            )
        _limit_basis(
            horizon["output_allowance_basis"],
            allowance,
            "horizon.output_allowance_basis",
            zero_kind=None,
        )
    if (
        completion_condition == "semantic_stop_or_allowance_per_turn"
        and allowance is None
    ):
        raise ValidationContractError(
            "semantic horizon completion requires a declared output "
            "allowance"
        )
    comparison = _object(document["comparison"], "comparison")
    _fields(
        comparison,
        {"output_mode", "state_mode"},
        "comparison",
    )
    output_mode = comparison["output_mode"]
    state_mode = comparison["state_mode"]
    if output_mode not in {"exact_digest", "fixture_semantics"}:
        raise ValidationContractError(
            "comparison.output_mode is unsupported"
        )
    if state_mode not in {"exact_digest", "trajectory_local"}:
        raise ValidationContractError(
            "comparison.state_mode is unsupported"
        )
    semantic_comparison = (
        output_mode == "fixture_semantics"
        or state_mode == "trajectory_local"
    )
    if semantic_comparison and (
        output_mode != "fixture_semantics"
        or state_mode != "trajectory_local"
        or completion_condition
        != "semantic_stop_or_allowance_per_turn"
        or regime["execution_scope"] != "whole_model"
        or document["kind"] not in {
            "free_running",
            "reasoning_conversation",
        }
    ):
        raise ValidationContractError(
            "trajectory-local fixture semantics are valid only for "
            "free-running whole-model conversations"
        )
    metrics = _sorted_unique_strings(
        document["metrics"],
        "metrics",
        nonempty=True,
    )
    if semantic_comparison and "semantic_consistency" not in metrics:
        raise ValidationContractError(
            "fixture-semantic comparison requires semantic_consistency"
        )
    if product_performance and (
        document["stage"] != "whole_model"
        or regime["execution_scope"] != "whole_model"
        or document["kind"] not in {
            "free_running",
            "reasoning_conversation",
        }
        or completion_condition
        != "semantic_stop_or_allowance_per_turn"
    ):
        raise ValidationContractError(
            "product performance requires a free-running whole-model "
            "semantic horizon"
        )
    if product_performance and (
        not isinstance(sampler, dict)
        or sampler.get("top_k") != 1
        or any(
            field in sampler
            for field in ("temperature", "top_p", "min_p")
        )
    ):
        raise ValidationContractError(
            "product performance requires deterministic top-k-one sampling"
        )
    expected = validation_check_id(document)
    if document["check_id"] != expected:
        raise ValidationContractError(
            f"validation check id must be {expected!r}"
        )


def validate_validation_plan(document: Json) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {
            "schema",
            "plan_id",
            "candidate_id",
            "source_contract_digests",
            "construction_record_digest",
            "requirements_digest",
            "behavioral_contract",
            "implementations",
            "matched_conditions",
            "matched_conditions_digest",
            "proofs",
            "checks",
            "coverage",
            "counterexamples",
        },
        "validation plan",
    )
    _schema(document, VALIDATION_PLAN_SCHEMA, "validation plan")
    _stable_id(document["plan_id"], "validation_plan", "plan_id")
    _stable_id(document["candidate_id"], "candidate", "candidate_id")
    _digests(
        document["source_contract_digests"],
        "source_contract_digests",
        nonempty=True,
    )
    _contract_digest(
        document["construction_record_digest"],
        "construction_record_digest",
    )
    _contract_digest(
        document["requirements_digest"],
        "requirements_digest",
    )
    validate_candidate_behavioral_contract(
        document["behavioral_contract"],
        "behavioral_contract",
    )
    implementations = _object(
        document["implementations"],
        "implementations",
    )
    _fields(
        implementations,
        set(_IMPLEMENTATION_ROLES),
        "implementations",
    )
    for role in _IMPLEMENTATION_ROLES:
        _implementation(
            implementations[role],
            f"implementations.{role}",
        )
    matched_conditions = _object(
        document["matched_conditions"],
        "matched_conditions",
    )
    validate_matched_conditions(matched_conditions)
    _contract_digest(
        document["matched_conditions_digest"],
        "matched_conditions_digest",
    )
    if (
        document["matched_conditions_digest"]
        != contract_digest(document["matched_conditions"])
    ):
        raise ValidationContractError(
            "matched conditions digest does not match validation conditions"
        )
    requirements = {
        "schema": VALIDATION_REQUIREMENTS_SCHEMA,
        "requirements_id": "",
        "candidate_id": document["candidate_id"],
        "source_contract_digests": document["source_contract_digests"],
        "proofs": document["proofs"],
        "checks": document["checks"],
        "coverage": document["coverage"],
        "counterexamples": document["counterexamples"],
    }
    requirements["requirements_id"] = validation_requirements_id(requirements)
    validate_validation_requirements(requirements)
    if document["requirements_digest"] != contract_digest(requirements):
        raise ValidationContractError(
            "validation plan requirements digest does not match its obligations"
        )
    proof_obligations = [
        proof["obligation"] for proof in document["proofs"]
    ]
    declared_obligations = document["behavioral_contract"][
        "proof_obligations"
    ]
    if proof_obligations != declared_obligations:
        raise ValidationContractError(
            "validation proof requirements do not cover the candidate obligations"
        )
    error_contract = document["behavioral_contract"]["error_contract"]
    check_metrics = {
        metric
        for check in document["checks"]
        for metric in check["metrics"]
    }
    if error_contract is not None:
        declared_metrics = {
            metric["name"] for metric in error_contract["metrics"]
        }
        if not check_metrics <= declared_metrics:
            raise ValidationContractError(
                "approximate validation checks use metrics outside the error contract"
            )
        required_coverage = {
            entry["kind"]
            for entry in document["coverage"]
            if entry["applicability"] == "required"
        }
        metric_coverage = {
            coverage
            for metric in error_contract["metrics"]
            for coverage in metric["coverage"]
        }
        if not required_coverage <= metric_coverage:
            raise ValidationContractError(
                "behavioral error contract does not cover every required behavior"
            )
    expected = validation_plan_id(document)
    if document["plan_id"] != expected:
        raise ValidationContractError(
            f"validation plan id must be {expected!r}"
        )


def validate_proof_result(document: Json) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {
            "schema",
            "proof_id",
            "plan_id",
            "candidate_id",
            "obligation",
            "verifier_id",
            "source_contract_digests",
            "construction_record_digest",
            "status",
            "facts",
            "artifacts",
            "diagnostics",
        },
        "proof result",
    )
    _schema(document, PROOF_RESULT_SCHEMA, "proof result")
    _stable_id(document["proof_id"], "proof_result", "proof_id")
    _stable_id(document["plan_id"], "validation_plan", "plan_id")
    _stable_id(document["candidate_id"], "candidate", "candidate_id")
    _text(document["obligation"], "obligation")
    _text(document["verifier_id"], "verifier_id")
    _digests(
        document["source_contract_digests"],
        "source_contract_digests",
        nonempty=True,
    )
    _contract_digest(
        document["construction_record_digest"],
        "construction_record_digest",
    )
    if document["status"] not in _PROOF_STATUSES:
        raise ValidationContractError("proof result status is unsupported")
    _object(document["facts"], "facts")
    artifacts = _artifact_refs(document["artifacts"], "artifacts")
    diagnostics = _string_list(document["diagnostics"], "diagnostics")
    if document["status"] == "proven" and not artifacts and not document["facts"]:
        raise ValidationContractError(
            "a proven obligation requires proof facts or artifacts"
        )
    if document["status"] != "proven" and not diagnostics:
        raise ValidationContractError(
            "an unproven obligation requires diagnostics"
        )
    expected = proof_result_id(document)
    if document["proof_id"] != expected:
        raise ValidationContractError(
            f"proof result id must be {expected!r}"
        )


def _horizon_completion(
    value: object,
    *,
    observed_steps: object,
    path: str,
) -> None:
    completion = _object(value, path)
    _fields(
        completion,
        {
            "condition",
            "satisfied",
            "observed_steps",
            "minimum_steps",
            "expected_turns",
            "completed_turns",
            "stop_reasons",
        },
        path,
    )
    condition = completion["condition"]
    if condition not in _HORIZON_COMPLETION_CONDITIONS:
        raise ValidationContractError(
            f"{path}.condition is unsupported"
        )
    if not isinstance(completion["satisfied"], bool):
        raise ValidationContractError(
            f"{path}.satisfied must be a boolean"
        )
    completed_steps = _nonnegative_integer(
        completion["observed_steps"],
        f"{path}.observed_steps",
    )
    if completed_steps != observed_steps:
        raise ValidationContractError(
            f"{path}.observed_steps does not match role-result steps"
        )
    stop_reasons = _string_list(
        completion["stop_reasons"],
        f"{path}.stop_reasons",
    )
    if condition == "minimum_steps":
        minimum_steps = _positive_integer(
            completion["minimum_steps"],
            f"{path}.minimum_steps",
        )
        if (
            completion["expected_turns"] is not None
            or completion["completed_turns"] is not None
            or stop_reasons
        ):
            raise ValidationContractError(
                f"{path} minimum-step evidence contains conversation fields"
            )
        satisfied = completed_steps >= minimum_steps
    elif condition == "all_fixture_turns":
        if completion["minimum_steps"] is not None:
            raise ValidationContractError(
                f"{path} fixture-turn evidence contains a minimum step count"
            )
        expected_turns = _positive_integer(
            completion["expected_turns"],
            f"{path}.expected_turns",
        )
        completed_turns = _nonnegative_integer(
            completion["completed_turns"],
            f"{path}.completed_turns",
        )
        if completed_turns > expected_turns:
            raise ValidationContractError(
                f"{path}.completed_turns exceeds expected turns"
            )
        if len(stop_reasons) != completed_turns or any(
            reason not in _FIXTURE_STOP_REASONS for reason in stop_reasons
        ):
            raise ValidationContractError(
                f"{path}.stop_reasons do not prove completed fixture turns"
            )
        satisfied = completed_turns == expected_turns
    else:
        if completion["minimum_steps"] is not None:
            raise ValidationContractError(
                f"{path} semantic evidence contains a minimum step count"
            )
        expected_turns = _positive_integer(
            completion["expected_turns"],
            f"{path}.expected_turns",
        )
        completed_turns = _nonnegative_integer(
            completion["completed_turns"],
            f"{path}.completed_turns",
        )
        if completed_turns > expected_turns:
            raise ValidationContractError(
                f"{path}.completed_turns exceeds expected turns"
            )
        if len(stop_reasons) != completed_turns or any(
            reason not in _SEMANTIC_STOP_REASONS
            for reason in stop_reasons
        ):
            raise ValidationContractError(
                f"{path}.stop_reasons do not prove completed conversation "
                "turns"
            )
        satisfied = completed_turns == expected_turns
    if completion["satisfied"] is not satisfied:
        raise ValidationContractError(
            f"{path}.satisfied contradicts its completion evidence"
        )


def validate_validation_role_result(document: Json) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {
            "schema",
            "result_id",
            "plan_id",
            "check_id",
            "stage",
            "seed",
            "role",
            "implementation_id",
            "status",
            "output_digest",
            "state_digest",
            "steps",
            "horizon_completion",
            "traces",
            "default_statistics",
            "diagnostics",
        },
        "validation role result",
    )
    _schema(
        document,
        VALIDATION_ROLE_RESULT_SCHEMA,
        "validation role result",
    )
    _stable_id(
        document["result_id"],
        "validation_role_result",
        "result_id",
    )
    _stable_id(document["plan_id"], "validation_plan", "plan_id")
    _stable_id(document["check_id"], "validation_check", "check_id")
    if document["stage"] not in VALIDATION_STAGES:
        raise ValidationContractError(
            "validation role-result stage is unsupported"
        )
    _u32(document["seed"], "seed")
    if document["role"] not in _IMPLEMENTATION_ROLES:
        raise ValidationContractError(
            "validation role-result role is unsupported"
        )
    _text(document["implementation_id"], "implementation_id")
    if document["status"] not in {"completed", "failed"}:
        raise ValidationContractError(
            "validation role-result status is unsupported"
        )
    for field in ("output_digest", "state_digest"):
        if document[field] is not None:
            _artifact_digest(document[field], field)
    _nonnegative_integer(document["steps"], "steps")
    _horizon_completion(
        document["horizon_completion"],
        observed_steps=document["steps"],
        path="horizon_completion",
    )
    _artifact_refs(document["traces"], "traces")
    if not _object(document["default_statistics"], "default_statistics"):
        raise ValidationContractError(
            "validation role result requires normal runtime statistics"
        )
    diagnostics = _string_list(document["diagnostics"], "diagnostics")
    if document["status"] == "failed" and not diagnostics:
        raise ValidationContractError(
            "failed validation role result requires diagnostics"
        )
    expected = validation_role_result_id(document)
    if document["result_id"] != expected:
        raise ValidationContractError(
            f"validation role result id must be {expected!r}"
        )


def validate_validation_observation(document: Json) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {
            "schema",
            "observation_id",
            "plan_id",
            "check_id",
            "stage",
            "seed",
            "status",
            "reference",
            "candidate",
            "metrics",
            "traces",
            "execution_statistics",
            "diagnostics",
        },
        "validation observation",
    )
    _schema(
        document,
        VALIDATION_OBSERVATION_SCHEMA,
        "validation observation",
    )
    _stable_id(
        document["observation_id"],
        "validation_observation",
        "observation_id",
    )
    _stable_id(document["plan_id"], "validation_plan", "plan_id")
    _stable_id(document["check_id"], "validation_check", "check_id")
    if document["stage"] not in VALIDATION_STAGES:
        raise ValidationContractError(
            "validation observation stage is unsupported"
        )
    _u32(document["seed"], "seed")
    if document["status"] not in {"completed", "failed"}:
        raise ValidationContractError(
            "validation observation status is unsupported"
        )
    for role in _IMPLEMENTATION_ROLES:
        result = _object(document[role], role)
        _fields(
            result,
            {
                "implementation_id",
                "output_digest",
                "state_digest",
                "steps",
                "horizon_completion",
            },
            role,
        )
        _text(result["implementation_id"], f"{role}.implementation_id")
        for field in ("output_digest", "state_digest"):
            if result[field] is not None:
                _artifact_digest(result[field], f"{role}.{field}")
        _nonnegative_integer(result["steps"], f"{role}.steps")
        _horizon_completion(
            result["horizon_completion"],
            observed_steps=result["steps"],
            path=f"{role}.horizon_completion",
        )
    metrics = _list(document["metrics"], "metrics")
    names: list[str] = []
    for index, raw_metric in enumerate(metrics):
        path = f"metrics[{index}]"
        metric = _object(raw_metric, path)
        _fields(
            metric,
            {
                "name",
                "reference_value",
                "candidate_value",
                "error",
                "unit",
            },
            path,
        )
        names.append(_text(metric["name"], f"{path}.name"))
        for field in ("reference_value", "candidate_value"):
            _finite_number(metric[field], f"{path}.{field}")
        _finite_nonnegative(metric["error"], f"{path}.error")
        _text(metric["unit"], f"{path}.unit")
    if names != sorted(set(names)):
        raise ValidationContractError(
            "validation observation metrics must be sorted and unique"
        )
    traces = _object(document["traces"], "traces")
    _fields(traces, set(_IMPLEMENTATION_ROLES), "traces")
    for role in _IMPLEMENTATION_ROLES:
        _artifact_refs(traces[role], f"traces.{role}")
    statistics = _object(
        document["execution_statistics"],
        "execution_statistics",
    )
    _fields(
        statistics,
        set(_IMPLEMENTATION_ROLES),
        "execution_statistics",
    )
    for role in _IMPLEMENTATION_ROLES:
        if not _object(
            statistics[role],
            f"execution_statistics.{role}",
        ):
            raise ValidationContractError(
                f"execution_statistics.{role} must not be empty"
            )
    diagnostics = _string_list(document["diagnostics"], "diagnostics")
    if document["status"] == "failed" and not diagnostics:
        raise ValidationContractError(
            "failed validation observation requires diagnostics"
        )
    expected = validation_observation_id(document)
    if document["observation_id"] != expected:
        raise ValidationContractError(
            f"validation observation id must be {expected!r}"
        )


def validate_validation_residency_event(document: Json) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {
            "schema",
            "event_id",
            "plan_id",
            "stage",
            "check_id",
            "seed",
            "role",
            "implementation_id",
            "block_index",
            "action",
            "duration_ns",
            "device_state_before_digest",
            "device_state_after_digest",
            "released",
            "default_statistics",
        },
        "validation residency event",
    )
    _schema(
        document,
        VALIDATION_RESIDENCY_EVENT_SCHEMA,
        "validation residency event",
    )
    _stable_id(
        document["event_id"],
        "validation_residency",
        "event_id",
    )
    _stable_id(document["plan_id"], "validation_plan", "plan_id")
    if document["stage"] not in VALIDATION_STAGES:
        raise ValidationContractError(
            "validation residency event stage is unsupported"
        )
    _stable_id(document["check_id"], "validation_check", "check_id")
    _u32(document["seed"], "seed")
    if document["role"] not in _IMPLEMENTATION_ROLES:
        raise ValidationContractError(
            "validation residency role is unsupported"
        )
    _text(document["implementation_id"], "implementation_id")
    _nonnegative_integer(document["block_index"], "block_index")
    if document["action"] not in {"mount", "unmount"}:
        raise ValidationContractError(
            "validation residency event action is unsupported"
        )
    _positive_integer(document["duration_ns"], "duration_ns")
    for field in (
        "device_state_before_digest",
        "device_state_after_digest",
    ):
        _device_state_digest(document[field], field)
    if not isinstance(document["released"], bool):
        raise ValidationContractError(
            "validation residency released must be boolean"
        )
    if (
        document["action"] == "mount"
        and document["released"]
    ) or (
        document["action"] == "unmount"
        and not document["released"]
    ):
        raise ValidationContractError(
            "validation residency released state contradicts its action"
        )
    _object(document["default_statistics"], "default_statistics")
    expected = validation_residency_event_id(document)
    if document["event_id"] != expected:
        raise ValidationContractError(
            f"validation residency event id must be {expected!r}"
        )


def validate_validation_run(document: Json) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {
            "schema",
            "run_id",
            "plan_id",
            "stage",
            "status",
            "execution_order",
            "observations",
            "residency_events",
            "host_elapsed_ns",
            "diagnostics",
        },
        "validation run",
    )
    _schema(document, VALIDATION_RUN_SCHEMA, "validation run")
    _stable_id(document["run_id"], "validation_run", "run_id")
    _stable_id(document["plan_id"], "validation_plan", "plan_id")
    if document["stage"] not in VALIDATION_STAGES:
        raise ValidationContractError("validation run stage is unsupported")
    if document["status"] not in _RUN_STATUSES:
        raise ValidationContractError("validation run status is unsupported")
    observations = _list(document["observations"], "observations")
    by_id: dict[str, Json] = {}
    for index, raw_observation in enumerate(observations):
        observation = _object(
            raw_observation,
            f"observations[{index}]",
        )
        validate_validation_observation(observation)
        observation_id = str(observation["observation_id"])
        if (
            observation["plan_id"] != document["plan_id"]
            or observation["stage"] != document["stage"]
            or observation_id in by_id
        ):
            raise ValidationContractError(
                "validation run observations do not match their run"
            )
        by_id[observation_id] = observation
    order = _string_list(document["execution_order"], "execution_order")
    if order != list(by_id):
        raise ValidationContractError(
            "validation run execution order must match observations exactly"
        )
    events = _list(document["residency_events"], "residency_events")
    if len(events) != 4 * len(observations):
        raise ValidationContractError(
            "validation run requires a mount and unmount for each role result"
        )
    parsed_events = []
    for index, raw_event in enumerate(events):
        event = _object(raw_event, f"residency_events[{index}]")
        validate_validation_residency_event(event)
        if (
            event["plan_id"] != document["plan_id"]
            or event["stage"] != document["stage"]
        ):
            raise ValidationContractError(
                "validation residency event does not match its run"
            )
        parsed_events.append(event)
    expected_residencies = {
        (
            observation["check_id"],
            observation["seed"],
            role,
            observation[role]["implementation_id"],
        )
        for observation in observations
        for role in _IMPLEMENTATION_ROLES
    }
    observed_residencies: set[tuple[str, int, str, str]] = set()
    capacity_reservation_digest: str | None = None
    for block_index, event_index in enumerate(
        range(0, len(parsed_events), 2)
    ):
        mount = parsed_events[event_index]
        unmount = parsed_events[event_index + 1]
        identity = (
            mount["check_id"],
            mount["seed"],
            mount["role"],
            mount["implementation_id"],
        )
        if (
            identity not in expected_residencies
            or identity in observed_residencies
            or any(
                unmount[field] != mount[field]
                for field in (
                    "check_id",
                    "seed",
                    "role",
                    "implementation_id",
                )
            )
            or mount["block_index"] != block_index
            or unmount["block_index"] != block_index
        ):
            raise ValidationContractError(
                "validation residency identity does not match execution"
            )
        observed_residencies.add(identity)
        if (
            mount["action"] != "mount"
            or unmount["action"] != "unmount"
            or mount["device_state_after_digest"]
            != unmount["device_state_before_digest"]
            or mount["device_state_before_digest"]
            != unmount["device_state_after_digest"]
        ):
            raise ValidationContractError(
                "validation role residency does not prove complete release"
            )
        if capacity_reservation_digest is None:
            capacity_reservation_digest = mount["device_state_before_digest"]
        elif mount["device_state_before_digest"] != capacity_reservation_digest:
            raise ValidationContractError(
                "validation role mounts do not share one capacity reservation"
            )
    if observed_residencies != expected_residencies:
        raise ValidationContractError(
            "validation run is missing required role residency evidence"
        )
    elapsed = _list(document["host_elapsed_ns"], "host_elapsed_ns")
    if len(elapsed) != len(observations):
        raise ValidationContractError(
            "validation host timing count does not match observations"
        )
    elapsed_ids: list[str] = []
    for index, raw_elapsed in enumerate(elapsed):
        path = f"host_elapsed_ns[{index}]"
        timing = _object(raw_elapsed, path)
        _fields(timing, {"observation_id", "duration_ns"}, path)
        elapsed_ids.append(
            _stable_id(
                timing["observation_id"],
                "validation_observation",
                f"{path}.observation_id",
            )
        )
        _positive_integer(timing["duration_ns"], f"{path}.duration_ns")
    if elapsed_ids != order:
        raise ValidationContractError(
            "validation host timings must follow execution order"
        )
    diagnostics = _string_list(document["diagnostics"], "diagnostics")
    if document["status"] == "completed":
        if not observations or any(
            observation["status"] != "completed"
            for observation in observations
        ):
            raise ValidationContractError(
                "completed validation run requires completed observations"
            )
    elif not diagnostics:
        raise ValidationContractError(
            "incomplete validation run requires diagnostics"
        )
    expected = validation_run_id(document)
    if document["run_id"] != expected:
        raise ValidationContractError(
            f"validation run id must be {expected!r}"
        )


def validate_prebenchmark_record(document: Json) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {
            "schema",
            "prebenchmark_id",
            "candidate_id",
            "source_contract_digests",
            "validation_plan_digest",
            "construction_record_digest",
            "static_validation",
            "proof_results",
            "sanity_run_digest",
            "stages",
            "counterexamples",
            "status",
        },
        "prebenchmark record",
    )
    _schema(
        document,
        PREBENCHMARK_RECORD_SCHEMA,
        "prebenchmark record",
    )
    _stable_id(
        document["prebenchmark_id"],
        "prebenchmark_validation",
        "prebenchmark_id",
    )
    _stable_id(document["candidate_id"], "candidate", "candidate_id")
    _digests(
        document["source_contract_digests"],
        "source_contract_digests",
        nonempty=True,
    )
    _contract_digest(
        document["validation_plan_digest"],
        "validation_plan_digest",
    )
    _contract_digest(
        document["construction_record_digest"],
        "construction_record_digest",
    )
    static = _object(document["static_validation"], "static_validation")
    _fields(
        static,
        {"status", "staged_integrity_digest", "artifact_count"},
        "static_validation",
    )
    if static["status"] not in {"passed", "failed"}:
        raise ValidationContractError(
            "static validation status is unsupported"
        )
    if static["staged_integrity_digest"] is not None:
        _artifact_digest(
            static["staged_integrity_digest"],
            "static_validation.staged_integrity_digest",
        )
    _nonnegative_integer(
        static["artifact_count"],
        "static_validation.artifact_count",
    )
    proof_results = _list(document["proof_results"], "proof_results")
    obligations: list[str] = []
    for index, raw_proof in enumerate(proof_results):
        proof = _object(raw_proof, f"proof_results[{index}]")
        validate_proof_result(proof)
        obligations.append(str(proof["obligation"]))
    if obligations != sorted(set(obligations)):
        raise ValidationContractError(
            "prebenchmark proof results must be sorted by obligation"
        )
    if document["sanity_run_digest"] is not None:
        _contract_digest(
            document["sanity_run_digest"],
            "sanity_run_digest",
        )
    stages = _stages(
        document["stages"],
        VALIDATION_FUNNEL_STAGE_NAMES[:3],
    )
    _artifact_refs(document["counterexamples"], "counterexamples")
    if document["status"] not in {"passed", "failed"}:
        raise ValidationContractError(
            "prebenchmark record status is unsupported"
        )
    if document["status"] == "passed" and any(
        stage["status"] not in {"passed", "not_applicable"}
        for stage in stages
    ):
        raise ValidationContractError(
            "passed prebenchmark record contains an incomplete stage"
        )
    expected = prebenchmark_record_id(document)
    if document["prebenchmark_id"] != expected:
        raise ValidationContractError(
            f"prebenchmark record id must be {expected!r}"
        )


def validate_validation_record(document: Json) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {
            "schema",
            "validation_id",
            "candidate_id",
            "source_contract_digests",
            "behavioral_contract",
            "validation_plan_digest",
            "construction_record_digest",
            "prebenchmark_record_digest",
            "benchmark_record_digest",
            "runs",
            "stages",
            "counterexamples",
            "status",
        },
        "validation record",
    )
    _stable_id(document["validation_id"], "validation", "validation_id")
    _stable_id(document["candidate_id"], "candidate", "candidate_id")
    _digests(
        document["source_contract_digests"],
        "source_contract_digests",
        nonempty=True,
    )
    validate_candidate_behavioral_contract(
        document["behavioral_contract"],
        "behavioral_contract",
    )
    for field in (
        "validation_plan_digest",
        "construction_record_digest",
        "prebenchmark_record_digest",
        "benchmark_record_digest",
    ):
        _contract_digest(document[field], field)
    runs = _list(document["runs"], "runs")
    run_stages: list[str] = []
    for index, raw_run in enumerate(runs):
        path = f"runs[{index}]"
        run = _object(raw_run, path)
        _fields(run, {"stage", "run_digest"}, path)
        if run["stage"] not in {"full_local", "whole_model"}:
            raise ValidationContractError(f"{path}.stage is unsupported")
        run_stages.append(str(run["stage"]))
        _contract_digest(run["run_digest"], f"{path}.run_digest")
    if run_stages != sorted(set(run_stages), key=("full_local", "whole_model").index):
        raise ValidationContractError(
            "validation record runs must be unique and in funnel order"
        )
    stages = _stages(
        document["stages"],
        VALIDATION_FUNNEL_STAGE_NAMES,
    )
    _artifact_refs(document["counterexamples"], "counterexamples")
    if document["status"] not in {"passed", "failed"}:
        raise ValidationContractError(
            "validation record status is unsupported"
        )
    if document["status"] == "passed" and any(
        stage["status"] not in {"passed", "not_applicable"}
        for stage in stages
    ):
        raise ValidationContractError(
            "passed validation record contains an incomplete stage"
        )
    expected = validation_record_id(document)
    if document["validation_id"] != expected:
        raise ValidationContractError(
            f"validation record id must be {expected!r}"
        )


def validate_candidate_behavioral_contract(
    value: object,
    path: str,
) -> Json:
    behavioral = _object(value, path)
    _fields(
        behavioral,
        {"mode", "proof_obligations", "error_contract"},
        path,
    )
    if behavioral["mode"] not in {"exact", "approximate"}:
        raise ValidationContractError(f"{path}.mode is unsupported")
    _sorted_unique_strings(
        behavioral["proof_obligations"],
        f"{path}.proof_obligations",
        nonempty=behavioral["mode"] == "exact",
    )
    if behavioral["mode"] == "exact":
        if behavioral["error_contract"] is not None:
            raise ValidationContractError(
                f"{path} exact mode cannot declare an error contract"
            )
    else:
        validate_behavioral_error_contract(
            _object(
                behavioral["error_contract"],
                f"{path}.error_contract",
            )
        )
    return behavioral


def _stages(value: object, expected_names: tuple[str, ...]) -> list[Json]:
    stages = _list(value, "stages")
    names: list[str] = []
    for index, raw_stage in enumerate(stages):
        path = f"stages[{index}]"
        stage = _object(raw_stage, path)
        _fields(
            stage,
            {
                "name",
                "status",
                "evidence_digests",
                "metrics",
                "artifacts",
                "reason",
            },
            path,
        )
        names.append(_text(stage["name"], f"{path}.name"))
        if stage["status"] not in _STAGE_STATUSES:
            raise ValidationContractError(
                f"{path}.status is unsupported"
            )
        _digests(
            stage["evidence_digests"],
            f"{path}.evidence_digests",
        )
        _object(stage["metrics"], f"{path}.metrics")
        _artifact_refs(stage["artifacts"], f"{path}.artifacts")
        if stage["status"] in {"failed", "not_applicable", "not_run"}:
            _text(stage["reason"], f"{path}.reason")
        elif stage["reason"] is not None:
            raise ValidationContractError(
                f"{path}.reason must be null for a passed stage"
            )
    if names != list(expected_names):
        raise ValidationContractError(
            "validation stages are missing or out of funnel order"
        )
    return stages


def _implementation(value: object, path: str) -> Json:
    implementation = _object(value, path)
    _fields(
        implementation,
        {"implementation_id", "contract_digest", "artifact_refs"},
        path,
    )
    _text(
        implementation["implementation_id"],
        f"{path}.implementation_id",
    )
    _contract_digest(
        implementation["contract_digest"],
        f"{path}.contract_digest",
    )
    _artifact_refs(
        implementation["artifact_refs"],
        f"{path}.artifact_refs",
    )
    return implementation


def _content_id(prefix: str, field: str, document: Json) -> str:
    unsigned = deepcopy(document)
    unsigned[field] = ""
    return stable_contract_id(prefix, unsigned)


def _fields(value: object, expected: set[str], path: str) -> Json:
    document = _object(value, path)
    actual = set(document)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ValidationContractError(f"{path} is missing fields {missing}")
    if unknown:
        raise ValidationContractError(f"{path} has unknown fields {unknown}")
    return document


def _schema(document: Json, expected: str, path: str) -> None:
    if document["schema"] != expected:
        raise ValidationContractError(
            f"{path} schema must be {expected!r}"
        )


def _object(value: object, path: str) -> Json:
    if not isinstance(value, dict):
        raise ValidationContractError(f"{path} must be an object")
    return value


def _list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationContractError(f"{path} must be a list")
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationContractError(f"{path} must be a non-empty string")
    return value


def _string_list(value: object, path: str) -> list[str]:
    values = _list(value, path)
    if not all(isinstance(item, str) and item for item in values):
        raise ValidationContractError(
            f"{path} must contain non-empty strings"
        )
    return values


def _sorted_unique_strings(
    value: object,
    path: str,
    *,
    nonempty: bool = False,
) -> list[str]:
    values = _string_list(value, path)
    if nonempty and not values:
        raise ValidationContractError(f"{path} must not be empty")
    if values != sorted(set(values)):
        raise ValidationContractError(
            f"{path} must be sorted and unique"
        )
    return values


def _stable_id(value: object, prefix: str, path: str) -> str:
    text = _text(value, path)
    expected_prefix = f"{prefix}_"
    if (
        not text.startswith(expected_prefix)
        or len(text) != len(expected_prefix) + 32
        or any(
            character not in "0123456789abcdef"
            for character in text.removeprefix(expected_prefix)
        )
    ):
        raise ValidationContractError(
            f"{path} must be a stable {prefix!r} identity"
        )
    return text


def _digests(
    value: object,
    path: str,
    *,
    nonempty: bool = False,
) -> list[str]:
    values = _sorted_unique_strings(value, path, nonempty=nonempty)
    for index, digest in enumerate(values):
        _contract_digest(digest, f"{path}[{index}]")
    return values


def _contract_digest(value: object, path: str) -> str:
    text = _text(value, path)
    suffix = text.removeprefix(_CONTRACT_DIGEST_PREFIX)
    if (
        not text.startswith(_CONTRACT_DIGEST_PREFIX)
        or len(suffix) != 64
        or any(character not in "0123456789abcdef" for character in suffix)
    ):
        raise ValidationContractError(
            f"{path} must be a canonical contract digest"
        )
    return text


def _artifact_digest(value: object, path: str) -> str:
    text = _text(value, path)
    suffix = text.removeprefix(_ARTIFACT_DIGEST_PREFIX)
    if (
        not text.startswith(_ARTIFACT_DIGEST_PREFIX)
        or len(suffix) != 64
        or any(character not in "0123456789abcdef" for character in suffix)
    ):
        raise ValidationContractError(
            f"{path} must be an artifact digest"
        )
    return text


def _device_state_digest(value: object, path: str) -> str:
    text = _text(value, path)
    prefix = f"{DEVICE_STATE_DIGEST_SCHEMA}:"
    suffix = text.removeprefix(prefix)
    if (
        not text.startswith(prefix)
        or len(suffix) != 64
        or any(character not in "0123456789abcdef" for character in suffix)
    ):
        raise ValidationContractError(
            f"{path} must be a device-state digest"
        )
    return text


def _artifact_ref(value: object, path: str) -> Json:
    reference = _object(value, path)
    _fields(reference, {"path", "digest"}, path)
    relative = _text(reference["path"], f"{path}.path")
    parsed = PurePosixPath(relative)
    if (
        parsed.is_absolute()
        or "." in parsed.parts
        or ".." in parsed.parts
        or parsed.as_posix() != relative
    ):
        raise ValidationContractError(
            f"{path}.path must be a normalized relative path"
        )
    _artifact_digest(reference["digest"], f"{path}.digest")
    return reference


def _artifact_refs(value: object, path: str) -> list[Json]:
    references = _list(value, path)
    paths: list[str] = []
    for index, reference in enumerate(references):
        parsed = _artifact_ref(reference, f"{path}[{index}]")
        paths.append(str(parsed["path"]))
    if paths != sorted(set(paths)):
        raise ValidationContractError(
            f"{path} must be sorted and unique by path"
        )
    return references


def _limit_basis(
    value: object,
    limit: int,
    path: str,
    *,
    zero_kind: str | None,
) -> Json:
    basis = _object(value, path)
    if limit == 0 and zero_kind is not None:
        _fields(basis, {"kind"}, path)
        if basis["kind"] != zero_kind:
            raise ValidationContractError(
                f"{path}.kind must be {zero_kind!r} for a zero limit"
            )
        return basis
    _fields(
        basis,
        {"kind", "artifact", "json_pointer", "declared_limit"},
        path,
    )
    if basis["kind"] != "declared_model_limit":
        raise ValidationContractError(
            f"{path}.kind must bind the source model's declared limit"
        )
    _artifact_ref(basis["artifact"], f"{path}.artifact")
    pointer = _text(basis["json_pointer"], f"{path}.json_pointer")
    if not pointer.startswith("/"):
        raise ValidationContractError(
            f"{path}.json_pointer must start with '/'"
        )
    for segment in pointer.split("/")[1:]:
        index = 0
        while index < len(segment):
            if segment[index] == "~" and (
                index + 1 == len(segment)
                or segment[index + 1] not in {"0", "1"}
            ):
                raise ValidationContractError(
                    f"{path}.json_pointer contains an invalid escape"
                )
            index += 2 if segment[index] == "~" else 1
    if basis["declared_limit"] != limit:
        raise ValidationContractError(
            f"{path}.declared_limit does not match the validation limit"
        )
    return basis


def _finite_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationContractError(f"{path} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationContractError(f"{path} must be finite")
    return number


def _finite_nonnegative(value: object, path: str) -> float:
    number = _finite_number(value, path)
    if number < 0:
        raise ValidationContractError(f"{path} must be non-negative")
    return number


def _finite_positive(value: object, path: str) -> float:
    number = _finite_number(value, path)
    if number <= 0:
        raise ValidationContractError(f"{path} must be positive")
    return number


def _nonnegative_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationContractError(
            f"{path} must be a non-negative integer"
        )
    return value


def _positive_integer(value: object, path: str) -> int:
    result = _nonnegative_integer(value, path)
    if result == 0:
        raise ValidationContractError(
            f"{path} must be a positive integer"
        )
    return result


def _u32(value: object, path: str) -> int:
    result = _nonnegative_integer(value, path)
    if result > 0xFFFF_FFFF:
        raise ValidationContractError(f"{path} must fit in U32")
    return result
