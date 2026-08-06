from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

from nerve.compilation import Json
from nerve.representation_optimizer.contracts import (
    ContractValidationError,
    canonical_json_bytes,
    stable_contract_id,
)


RUNTIME_IMPLEMENTATION_PREDICATE_SCHEMA = (
    "nerve.optimizer.runtime_implementation_predicate.v6"
)
PROMOTION_DECISION_SCHEMA = "nerve.optimizer.promotion_decision.v2"
IMPLEMENTATION_REGISTRY_SCHEMA = "nerve.optimizer.implementation_registry.v1"

_CONTRACT_DIGEST_PREFIX = "nerve.optimizer.canonical_json_sha256.v1:"
_ARTIFACT_DIGEST_PREFIX = "nerve.optimizer.artifact_sha256.v1:"
_EXECUTION_PHASES = (
    "component",
    "decode",
    "mixed",
    "prefill",
    "state_transition",
)
_PLACEMENT_MODES = ("distributed", "either", "local")


class PromotionContractError(ContractValidationError):
    """A promotion, target predicate, or implementation registry is invalid."""


@dataclass(frozen=True)
class RuntimeImplementationPredicate:
    _document: Json

    @classmethod
    def from_json(
        cls,
        document: Json,
    ) -> RuntimeImplementationPredicate:
        normalized = deepcopy(document)
        validate_runtime_implementation_predicate(normalized)
        return cls(normalized)

    @property
    def predicate_id(self) -> str:
        return str(self._document["predicate_id"])

    def to_json(self) -> Json:
        return deepcopy(self._document)


@dataclass(frozen=True)
class PromotionDecision:
    _document: Json

    @classmethod
    def from_json(cls, document: Json) -> PromotionDecision:
        normalized = deepcopy(document)
        validate_promotion_decision(normalized)
        return cls(normalized)

    @property
    def promotion_id(self) -> str:
        return str(self._document["promotion_id"])

    @property
    def candidate_id(self) -> str:
        return str(self._document["candidate_id"])

    @property
    def implementation_id(self) -> str:
        return str(self._document["implementation_id"])

    def to_json(self) -> Json:
        return deepcopy(self._document)


@dataclass(frozen=True)
class ImplementationRegistry:
    _document: Json

    @classmethod
    def from_json(cls, document: Json) -> ImplementationRegistry:
        normalized = deepcopy(document)
        validate_implementation_registry(normalized)
        return cls(normalized)

    @property
    def registry_id(self) -> str:
        return str(self._document["registry_id"])

    @property
    def implementations(self) -> tuple[Json, ...]:
        return tuple(deepcopy(self._document["implementations"]))

    def to_json(self) -> Json:
        return deepcopy(self._document)


def runtime_predicate_id(document: Json) -> str:
    return _content_id(
        "runtime_predicate",
        "predicate_id",
        document,
    )


def promotion_decision_id(document: Json) -> str:
    return _content_id("promotion", "promotion_id", document)


def implementation_registry_id(document: Json) -> str:
    return _content_id(
        "implementation_registry",
        "registry_id",
        document,
    )


def implementation_id(
    candidate_id: str,
    runtime_predicate: RuntimeImplementationPredicate,
) -> str:
    return stable_contract_id(
        "implementation",
        candidate_id,
        runtime_predicate.to_json(),
    )


def create_runtime_implementation_predicate(
    *,
    measured_profile_ids: Iterable[str],
    capability_classes: Iterable[str],
    device_kinds: Iterable[str],
    apis: Iterable[str],
    required_processes: Iterable[str],
    required_features: Iterable[str],
    execution_phases: Iterable[str],
    alternative_execution_phases: Iterable[str],
    source_retained_execution_phases: Iterable[str],
    activation_batch_minimum: int,
    activation_batch_maximum: int,
    context_activations_minimum: int,
    context_activations_maximum: int,
    state_activations_minimum: int,
    state_activations_maximum: int,
    speculative_draft_token_counts: Iterable[int],
    residency_policies: Iterable[str],
    placement_mode: str,
    minimum_device_count: int,
    maximum_device_count: int,
    required_interconnects: Iterable[str],
) -> RuntimeImplementationPredicate:
    capability_classes = sorted(set(capability_classes))
    document = {
        "schema": RUNTIME_IMPLEMENTATION_PREDICATE_SCHEMA,
        "predicate_id": "",
        "hardware": {
            "measured_profile_ids": sorted(set(measured_profile_ids)),
            "capability_classes": capability_classes,
            "device_kinds": sorted(set(device_kinds)),
            "apis": sorted(set(apis)),
            "required_processes": sorted(set(required_processes)),
            "required_features": sorted(set(required_features)),
        },
        "execution": {
            "phases": sorted(set(execution_phases)),
            "alternative_phases": sorted(set(alternative_execution_phases)),
            "source_retained_phases": sorted(set(source_retained_execution_phases)),
            "activation_batch": {
                "minimum": activation_batch_minimum,
                "maximum": activation_batch_maximum,
            },
            "context_activations": {
                "minimum": context_activations_minimum,
                "maximum": context_activations_maximum,
            },
            "state_activations": {
                "minimum": state_activations_minimum,
                "maximum": state_activations_maximum,
            },
            "speculative_draft_token_counts": sorted(
                set(speculative_draft_token_counts)
            ),
            "residency_policies": sorted(set(residency_policies)),
        },
        "placement": {
            "mode": placement_mode,
            "minimum_device_count": minimum_device_count,
            "maximum_device_count": maximum_device_count,
            "required_interconnects": sorted(set(required_interconnects)),
        },
    }
    document["predicate_id"] = runtime_predicate_id(document)
    return RuntimeImplementationPredicate.from_json(document)


def create_empty_implementation_registry(
    *,
    package_id: str,
    exact_baseline: Json,
) -> ImplementationRegistry:
    document = {
        "schema": IMPLEMENTATION_REGISTRY_SCHEMA,
        "registry_id": "",
        "package_id": package_id,
        "exact_baseline": deepcopy(exact_baseline),
        "implementations": [],
    }
    document["registry_id"] = implementation_registry_id(document)
    return ImplementationRegistry.from_json(document)


def append_implementation_registry_entries(
    registry: ImplementationRegistry,
    entries: Iterable[Json],
) -> ImplementationRegistry:
    document = registry.to_json()
    existing = {
        entry["implementation_id"]: entry for entry in document["implementations"]
    }
    for raw_entry in entries:
        entry = deepcopy(raw_entry)
        validate_implementation_registry_entry(entry)
        identifier = str(entry["implementation_id"])
        previous = existing.get(identifier)
        if previous is not None and previous != entry:
            raise PromotionContractError(
                "implementation identity is already bound to different metadata"
            )
        existing[identifier] = entry
    document["implementations"] = [
        existing[identifier] for identifier in sorted(existing)
    ]
    document["registry_id"] = implementation_registry_id(document)
    return ImplementationRegistry.from_json(document)


def validate_runtime_implementation_predicate(document: Json) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {"schema", "predicate_id", "hardware", "execution", "placement"},
        "runtime implementation predicate",
    )
    _schema(
        document,
        RUNTIME_IMPLEMENTATION_PREDICATE_SCHEMA,
        "runtime implementation predicate",
    )
    _stable_id(document["predicate_id"], "runtime_predicate", "predicate_id")
    hardware = _object(document["hardware"], "hardware")
    _fields(
        hardware,
        {
            "measured_profile_ids",
            "capability_classes",
            "device_kinds",
            "apis",
            "required_processes",
            "required_features",
        },
        "hardware",
    )
    measured_profile_ids = _sorted_unique_strings(
        hardware["measured_profile_ids"],
        "hardware.measured_profile_ids",
        nonempty=True,
    )
    for index, profile_id in enumerate(measured_profile_ids):
        _stable_id(
            profile_id,
            "hardware_profile",
            f"hardware.measured_profile_ids[{index}]",
        )
    capability_classes = _sorted_unique_strings(
        hardware["capability_classes"],
        "hardware.capability_classes",
        nonempty=True,
    )
    for index, capability_class in enumerate(capability_classes):
        _stable_id(
            capability_class,
            "hardware_capability",
            f"hardware.capability_classes[{index}]",
        )
    _sorted_unique_strings(
        hardware["device_kinds"],
        "hardware.device_kinds",
        nonempty=True,
    )
    _sorted_unique_strings(
        hardware["apis"],
        "hardware.apis",
        nonempty=True,
    )
    _sorted_unique_strings(
        hardware["required_processes"],
        "hardware.required_processes",
    )
    _sorted_unique_strings(
        hardware["required_features"],
        "hardware.required_features",
    )

    execution = _object(document["execution"], "execution")
    _fields(
        execution,
        {
            "phases",
            "alternative_phases",
            "source_retained_phases",
            "activation_batch",
            "context_activations",
            "state_activations",
            "speculative_draft_token_counts",
            "residency_policies",
        },
        "execution",
    )
    phases = _sorted_unique_strings(
        execution["phases"],
        "execution.phases",
        nonempty=True,
    )
    if any(phase not in _EXECUTION_PHASES for phase in phases):
        raise PromotionContractError(
            "runtime predicate contains an unsupported execution phase"
        )
    alternative_phases = _sorted_unique_strings(
        execution["alternative_phases"],
        "execution.alternative_phases",
        nonempty=True,
    )
    source_retained_phases = _sorted_unique_strings(
        execution["source_retained_phases"],
        "execution.source_retained_phases",
    )
    if set(alternative_phases) & set(source_retained_phases) or set(
        alternative_phases
    ) | set(source_retained_phases) != set(phases):
        raise PromotionContractError(
            "runtime predicate must partition every execution phase into "
            "alternative or source-retained execution"
        )
    _inclusive_range(
        execution["activation_batch"], "execution.activation_batch", positive=True
    )
    _inclusive_range(
        execution["context_activations"],
        "execution.context_activations",
        positive=False,
    )
    _inclusive_range(
        execution["state_activations"],
        "execution.state_activations",
        positive=False,
    )
    speculative_draft_token_counts = _list(
        execution["speculative_draft_token_counts"],
        "execution.speculative_draft_token_counts",
    )
    if (
        not speculative_draft_token_counts
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in speculative_draft_token_counts
        )
        or speculative_draft_token_counts != sorted(set(speculative_draft_token_counts))
    ):
        raise PromotionContractError(
            "execution.speculative_draft_token_counts must contain sorted, "
            "unique non-negative integers"
        )
    residency_policies = _sorted_unique_strings(
        execution["residency_policies"],
        "execution.residency_policies",
        nonempty=True,
    )
    if any(
        policy not in {"demand_paged", "demand_retained", "eager"}
        for policy in residency_policies
    ):
        raise PromotionContractError(
            "execution.residency_policies contains an unsupported policy"
        )

    placement = _object(document["placement"], "placement")
    _fields(
        placement,
        {
            "mode",
            "minimum_device_count",
            "maximum_device_count",
            "required_interconnects",
        },
        "placement",
    )
    if placement["mode"] not in _PLACEMENT_MODES:
        raise PromotionContractError("runtime predicate placement mode is unsupported")
    minimum_devices = _positive_integer(
        placement["minimum_device_count"],
        "placement.minimum_device_count",
    )
    maximum_devices = _positive_integer(
        placement["maximum_device_count"],
        "placement.maximum_device_count",
    )
    if minimum_devices > maximum_devices:
        raise PromotionContractError("runtime predicate device-count range is inverted")
    _sorted_unique_strings(
        placement["required_interconnects"],
        "placement.required_interconnects",
    )
    if placement["mode"] == "local" and maximum_devices != 1:
        raise PromotionContractError(
            "local runtime predicate must use exactly one device"
        )
    if placement["mode"] == "distributed" and minimum_devices < 2:
        raise PromotionContractError(
            "distributed runtime predicate must require at least two devices"
        )
    expected = runtime_predicate_id(document)
    if document["predicate_id"] != expected:
        raise PromotionContractError(f"runtime predicate id must be {expected!r}")


def validate_promotion_decision(document: Json) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {
            "schema",
            "promotion_id",
            "candidate_id",
            "implementation_id",
            "scope_ids",
            "source_contract_digests",
            "candidate_contract_digest",
            "construction_record_digest",
            "prebenchmark_record_digest",
            "benchmark_record_digest",
            "validation_record_digest",
            "runtime_predicate",
            "artifact_integrity",
            "comparison",
            "provenance",
            "decision",
            "reason",
        },
        "promotion decision",
    )
    _schema(document, PROMOTION_DECISION_SCHEMA, "promotion decision")
    _stable_id(document["promotion_id"], "promotion", "promotion_id")
    _stable_id(document["candidate_id"], "candidate", "candidate_id")
    _stable_id(
        document["implementation_id"],
        "implementation",
        "implementation_id",
    )
    scope_ids = _unique_strings(
        document["scope_ids"],
        "scope_ids",
        nonempty=True,
    )
    for index, scope_id in enumerate(scope_ids):
        _stable_id(scope_id, "scope", f"scope_ids[{index}]")
    _contract_digests(
        document["source_contract_digests"],
        "source_contract_digests",
        nonempty=True,
    )
    for field in (
        "candidate_contract_digest",
        "construction_record_digest",
        "prebenchmark_record_digest",
        "benchmark_record_digest",
        "validation_record_digest",
    ):
        _contract_digest(document[field], field)
    predicate = RuntimeImplementationPredicate.from_json(
        _object(document["runtime_predicate"], "runtime_predicate")
    )
    if document["implementation_id"] != implementation_id(
        document["candidate_id"],
        predicate,
    ):
        raise PromotionContractError(
            "implementation id does not match candidate and runtime predicate"
        )
    integrity = _object(document["artifact_integrity"], "artifact_integrity")
    _fields(
        integrity,
        {"schema", "digest", "file_count"},
        "artifact_integrity",
    )
    _text(integrity["schema"], "artifact_integrity.schema")
    _artifact_digest(integrity["digest"], "artifact_integrity.digest")
    _positive_integer(
        integrity["file_count"],
        "artifact_integrity.file_count",
    )
    comparison = _object(document["comparison"], "comparison")
    _fields(
        comparison,
        {
            "exact_implementation_id",
            "exact_contract_digest",
            "benchmark_id",
            "benchmark_decision",
            "workloads",
            "validation_id",
            "validation_status",
            "behavioral_contract",
        },
        "comparison",
    )
    _text(
        comparison["exact_implementation_id"],
        "comparison.exact_implementation_id",
    )
    _contract_digest(
        comparison["exact_contract_digest"],
        "comparison.exact_contract_digest",
    )
    _stable_id(
        comparison["benchmark_id"],
        "benchmark",
        "comparison.benchmark_id",
    )
    if comparison["benchmark_decision"] != "materially_faster":
        raise PromotionContractError(
            "promotion comparison must contain a material benchmark win"
        )
    workloads = _list(comparison["workloads"], "comparison.workloads")
    if not workloads:
        raise PromotionContractError(
            "promotion comparison must retain per-regime benchmark evidence"
        )
    workload_ids = []
    for index, raw_workload in enumerate(workloads):
        path = f"comparison.workloads[{index}]"
        workload = _object(raw_workload, path)
        _fields(
            workload,
            {"workload_id", "decision", "paired"},
            path,
        )
        workload_ids.append(
            _stable_id(
                workload["workload_id"],
                "benchmark_workload",
                f"{path}.workload_id",
            )
        )
        if workload["decision"] != "materially_faster":
            raise PromotionContractError(
                "every promoted runtime regime must be materially faster"
            )
        _object(workload["paired"], f"{path}.paired")
    if len(workload_ids) != len(set(workload_ids)):
        raise PromotionContractError("promotion benchmark workloads must be unique")
    _stable_id(
        comparison["validation_id"],
        "validation",
        "comparison.validation_id",
    )
    if comparison["validation_status"] != "passed":
        raise PromotionContractError(
            "promotion comparison must contain passed behavioral validation"
        )
    _object(
        comparison["behavioral_contract"],
        "comparison.behavioral_contract",
    )
    provenance = _object(document["provenance"], "provenance")
    _fields(
        provenance,
        {
            "provider",
            "descriptor_id",
            "evidence_refs",
            "analysis_runs",
            "hardware_profiles",
            "representation_graph_digest",
            "target_lowering_digest",
            "relowering_request_digest",
        },
        "provenance",
    )
    _provider_identity(provenance["provider"], "provenance.provider")
    _stable_id(
        provenance["descriptor_id"],
        "representation_descriptor",
        "provenance.descriptor_id",
    )
    evidence_refs = _sorted_unique_strings(
        provenance["evidence_refs"],
        "provenance.evidence_refs",
        nonempty=True,
    )
    for index, evidence_ref in enumerate(evidence_refs):
        _stable_id(
            evidence_ref,
            "evidence",
            f"provenance.evidence_refs[{index}]",
        )
    analysis_runs = _list(
        provenance["analysis_runs"],
        "provenance.analysis_runs",
    )
    run_ids = []
    cited_evidence_ids = []
    for index, raw_run in enumerate(analysis_runs):
        path = f"provenance.analysis_runs[{index}]"
        run = _object(raw_run, path)
        _fields(
            run,
            {"run_id", "run_digest", "cited_evidence_ids"},
            path,
        )
        run_ids.append(_stable_id(run["run_id"], "analysis_run", f"{path}.run_id"))
        _contract_digest(run["run_digest"], f"{path}.run_digest")
        cited = _sorted_unique_strings(
            run["cited_evidence_ids"],
            f"{path}.cited_evidence_ids",
            nonempty=True,
        )
        for evidence_index, evidence_id in enumerate(cited):
            cited_evidence_ids.append(
                _stable_id(
                    evidence_id,
                    "evidence",
                    f"{path}.cited_evidence_ids[{evidence_index}]",
                )
            )
    if run_ids != sorted(set(run_ids)):
        raise PromotionContractError(
            "promotion analysis runs must be sorted and unique"
        )
    if sorted(cited_evidence_ids) != evidence_refs:
        raise PromotionContractError(
            "promotion analysis runs must cover every cited evidence record once"
        )
    hardware_profiles = _list(
        provenance["hardware_profiles"],
        "provenance.hardware_profiles",
    )
    profile_ids = []
    for index, raw_profile in enumerate(hardware_profiles):
        path = f"provenance.hardware_profiles[{index}]"
        profile = _object(raw_profile, path)
        _fields(profile, {"profile_id", "profile_digest"}, path)
        profile_ids.append(
            _stable_id(
                profile["profile_id"],
                "hardware_profile",
                f"{path}.profile_id",
            )
        )
        _contract_digest(
            profile["profile_digest"],
            f"{path}.profile_digest",
        )
    if profile_ids != sorted(set(profile_ids)) or not profile_ids:
        raise PromotionContractError(
            "promotion hardware profiles must be non-empty, sorted, and unique"
        )
    for field in (
        "representation_graph_digest",
        "target_lowering_digest",
        "relowering_request_digest",
    ):
        _contract_digest(provenance[field], f"provenance.{field}")
    if document["decision"] != "promote":
        raise PromotionContractError("published promotion decisions must be promote")
    _text(document["reason"], "reason")
    expected = promotion_decision_id(document)
    if document["promotion_id"] != expected:
        raise PromotionContractError(f"promotion id must be {expected!r}")


def validate_implementation_registry(document: Json) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {
            "schema",
            "registry_id",
            "package_id",
            "exact_baseline",
            "implementations",
        },
        "implementation registry",
    )
    _schema(
        document,
        IMPLEMENTATION_REGISTRY_SCHEMA,
        "implementation registry",
    )
    _stable_id(
        document["registry_id"],
        "implementation_registry",
        "registry_id",
    )
    _text(document["package_id"], "package_id")
    baseline = _object(document["exact_baseline"], "exact_baseline")
    _fields(
        baseline,
        {"artifact_ref", "contract_digest", "mutable"},
        "exact_baseline",
    )
    _safe_package_path(
        baseline["artifact_ref"],
        "exact_baseline.artifact_ref",
    )
    _contract_digest(
        baseline["contract_digest"],
        "exact_baseline.contract_digest",
    )
    if baseline["mutable"] is not False:
        raise PromotionContractError(
            "implementation registry exact baseline must be immutable"
        )
    implementations = _list(
        document["implementations"],
        "implementations",
    )
    identifiers = []
    for index, raw_entry in enumerate(implementations):
        entry = _object(raw_entry, f"implementations[{index}]")
        validate_implementation_registry_entry(entry)
        identifiers.append(str(entry["implementation_id"]))
    if identifiers != sorted(set(identifiers)):
        raise PromotionContractError(
            "implementation registry entries must be sorted and unique"
        )
    expected = implementation_registry_id(document)
    if document["registry_id"] != expected:
        raise PromotionContractError(f"implementation registry id must be {expected!r}")


def validate_implementation_registry_entry(document: Json) -> None:
    _fields(
        document,
        {
            "implementation_id",
            "candidate_id",
            "scope_ids",
            "source_contract_digests",
            "representation",
            "behavioral_contract",
            "runtime_predicate",
            "artifact_bundle",
            "evidence",
            "provenance",
            "comparison",
            "decision_reason",
        },
        "implementation registry entry",
    )
    _stable_id(
        document["implementation_id"],
        "implementation",
        "implementation_id",
    )
    _stable_id(document["candidate_id"], "candidate", "candidate_id")
    scope_ids = _unique_strings(
        document["scope_ids"],
        "scope_ids",
        nonempty=True,
    )
    for index, scope_id in enumerate(scope_ids):
        _stable_id(scope_id, "scope", f"scope_ids[{index}]")
    _contract_digests(
        document["source_contract_digests"],
        "source_contract_digests",
        nonempty=True,
    )
    _object(document["representation"], "representation")
    _object(document["behavioral_contract"], "behavioral_contract")
    predicate = RuntimeImplementationPredicate.from_json(
        _object(document["runtime_predicate"], "runtime_predicate")
    )
    if document["implementation_id"] != implementation_id(
        document["candidate_id"],
        predicate,
    ):
        raise PromotionContractError(
            "registry implementation id does not match its runtime predicate"
        )
    bundle = _object(document["artifact_bundle"], "artifact_bundle")
    _fields(
        bundle,
        {
            "root_ref",
            "candidate_integrity_ref",
            "mount_plan_ref",
            "candidate_integrity_digest",
            "artifact_count",
        },
        "artifact_bundle",
    )
    root_ref = _safe_package_path(
        bundle["root_ref"],
        "artifact_bundle.root_ref",
    )
    integrity_ref = _safe_package_path(
        bundle["candidate_integrity_ref"],
        "artifact_bundle.candidate_integrity_ref",
    )
    mount_plan_ref = _safe_package_path(
        bundle["mount_plan_ref"],
        "artifact_bundle.mount_plan_ref",
    )
    if not integrity_ref.startswith(f"{root_ref}/") or not mount_plan_ref.startswith(
        f"{root_ref}/"
    ):
        raise PromotionContractError(
            "candidate artifact references must stay inside their artifact bundle"
        )
    _artifact_digest(
        bundle["candidate_integrity_digest"],
        "artifact_bundle.candidate_integrity_digest",
    )
    _positive_integer(
        bundle["artifact_count"],
        "artifact_bundle.artifact_count",
    )
    evidence = _object(document["evidence"], "evidence")
    _fields(
        evidence,
        {
            "promotion_decision_ref",
            "candidate_contract_ref",
            "construction_record_ref",
            "prebenchmark_record_ref",
            "benchmark_record_ref",
            "validation_record_ref",
            "analysis_run_refs",
            "hardware_profile_refs",
        },
        "evidence",
    )
    for name, value in evidence.items():
        if name in {"analysis_run_refs", "hardware_profile_refs"}:
            continue
        _safe_package_path(value, f"evidence.{name}")
    analysis_run_refs = _list(
        evidence["analysis_run_refs"],
        "evidence.analysis_run_refs",
    )
    run_ids = []
    for index, raw_reference in enumerate(analysis_run_refs):
        path = f"evidence.analysis_run_refs[{index}]"
        reference = _object(raw_reference, path)
        _fields(reference, {"run_id", "artifact_ref"}, path)
        run_ids.append(
            _stable_id(
                reference["run_id"],
                "analysis_run",
                f"{path}.run_id",
            )
        )
        _safe_package_path(
            reference["artifact_ref"],
            f"{path}.artifact_ref",
        )
    if run_ids != sorted(set(run_ids)):
        raise PromotionContractError(
            "analysis run references must be sorted and unique"
        )
    hardware_profile_refs = _list(
        evidence["hardware_profile_refs"],
        "evidence.hardware_profile_refs",
    )
    profile_ids = []
    for index, raw_reference in enumerate(hardware_profile_refs):
        path = f"evidence.hardware_profile_refs[{index}]"
        reference = _object(raw_reference, path)
        _fields(reference, {"profile_id", "artifact_ref"}, path)
        profile_ids.append(
            _stable_id(
                reference["profile_id"],
                "hardware_profile",
                f"{path}.profile_id",
            )
        )
        _safe_package_path(
            reference["artifact_ref"],
            f"{path}.artifact_ref",
        )
    if profile_ids != sorted(set(profile_ids)) or not profile_ids:
        raise PromotionContractError(
            "hardware profile references must be non-empty, sorted, and unique"
        )
    _object(document["provenance"], "provenance")
    _object(document["comparison"], "comparison")
    _text(document["decision_reason"], "decision_reason")


def _content_id(prefix: str, id_field: str, document: Json) -> str:
    unsigned = deepcopy(document)
    unsigned.pop(id_field, None)
    return stable_contract_id(prefix, unsigned)


def _inclusive_range(value: object, path: str, *, positive: bool) -> None:
    range_document = _object(value, path)
    _fields(range_document, {"minimum", "maximum"}, path)
    validator = _positive_integer if positive else _nonnegative_integer
    minimum = validator(range_document["minimum"], f"{path}.minimum")
    maximum = validator(range_document["maximum"], f"{path}.maximum")
    if minimum > maximum:
        raise PromotionContractError(f"{path} is inverted")


def _provider_identity(value: object, path: str) -> None:
    identity = _object(value, path)
    _fields(identity, {"id", "version"}, path)
    _text(identity["id"], f"{path}.id")
    _text(identity["version"], f"{path}.version")


def _fields(document: Json, expected: set[str], path: str) -> None:
    actual = set(document)
    if actual != expected:
        raise PromotionContractError(
            f"{path} fields differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _schema(document: Json, expected: str, path: str) -> None:
    if document.get("schema") != expected:
        raise PromotionContractError(f"{path} schema is unsupported")


def _object(value: object, path: str) -> Json:
    if not isinstance(value, dict):
        raise PromotionContractError(f"{path} must be an object")
    return value


def _list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise PromotionContractError(f"{path} must be a list")
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise PromotionContractError(f"{path} must be a non-empty string")
    return value


def _nonnegative_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PromotionContractError(f"{path} must be a non-negative integer")
    return value


def _positive_integer(value: object, path: str) -> int:
    parsed = _nonnegative_integer(value, path)
    if parsed == 0:
        raise PromotionContractError(f"{path} must be positive")
    return parsed


def _sorted_unique_strings(
    value: object,
    path: str,
    *,
    nonempty: bool = False,
) -> list[str]:
    values = _list(value, path)
    if any(not isinstance(item, str) or not item for item in values):
        raise PromotionContractError(f"{path} must contain non-empty strings")
    if values != sorted(set(values)):
        raise PromotionContractError(f"{path} must be sorted and unique")
    if nonempty and not values:
        raise PromotionContractError(f"{path} must not be empty")
    return values


def _unique_strings(
    value: object,
    path: str,
    *,
    nonempty: bool = False,
) -> list[str]:
    values = _list(value, path)
    if any(not isinstance(item, str) or not item for item in values):
        raise PromotionContractError(f"{path} must contain non-empty strings")
    if len(values) != len(set(values)):
        raise PromotionContractError(f"{path} must be unique")
    if nonempty and not values:
        raise PromotionContractError(f"{path} must not be empty")
    return values


def _stable_id(value: object, prefix: str, path: str) -> str:
    identifier = _text(value, path)
    expected_prefix = f"{prefix}_"
    suffix = identifier.removeprefix(expected_prefix)
    if (
        not identifier.startswith(expected_prefix)
        or len(suffix) != 32
        or any(character not in "0123456789abcdef" for character in suffix)
    ):
        raise PromotionContractError(f"{path} must be a stable {prefix!r} identifier")
    return identifier


def _contract_digest(value: object, path: str) -> str:
    return _digest(value, path, _CONTRACT_DIGEST_PREFIX)


def _artifact_digest(value: object, path: str) -> str:
    return _digest(value, path, _ARTIFACT_DIGEST_PREFIX)


def _digest(value: object, path: str, prefix: str) -> str:
    digest = _text(value, path)
    suffix = digest.removeprefix(prefix)
    if (
        not digest.startswith(prefix)
        or len(suffix) != 64
        or any(character not in "0123456789abcdef" for character in suffix)
    ):
        raise PromotionContractError(f"{path} has an invalid digest")
    return digest


def _contract_digests(
    value: object,
    path: str,
    *,
    nonempty: bool,
) -> list[str]:
    digests = _unique_strings(value, path, nonempty=nonempty)
    for index, digest in enumerate(digests):
        _contract_digest(digest, f"{path}[{index}]")
    return digests


def _safe_package_path(value: object, path: str) -> str:
    text = _text(value, path)
    from pathlib import PurePosixPath

    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or "." in relative.parts
        or ".." in relative.parts
        or relative.as_posix() != text
    ):
        raise PromotionContractError(
            f"{path} must be a canonical package-relative path"
        )
    return text
