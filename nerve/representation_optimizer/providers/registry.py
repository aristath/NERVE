from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

from nerve.compilation import Json
from nerve.representation_optimizer.benchmarking.contracts import (
    BenchmarkWorkload,
)
from nerve.representation_optimizer.contracts import (
    REPRESENTATION_CANDIDATE_SCHEMA,
    ContractDocument,
    ContractValidationError,
    canonical_json_bytes,
    representation_candidate_equivalence_key,
)
from nerve.representation_optimizer.descriptor_registry import (
    RepresentationDescriptorRegistry,
)
from nerve.representation_optimizer.mounting import RuntimeMountPlan
from nerve.representation_optimizer.providers.protocol import (
    RepresentationProvider,
)
from nerve.representation_optimizer.providers.types import (
    EvidenceAssessment,
    MatchAssessment,
    ProviderCandidatePlan,
    ProviderContext,
    ProviderEvaluation,
    ProviderIdentity,
    ProviderProblem,
    ProviderRegistryReport,
    StaticEstimate,
)
from nerve.representation_optimizer.representation_ir import (
    RepresentationGraphDocument,
)
from nerve.representation_optimizer.staging.contracts import CandidateBuildPlan
from nerve.representation_optimizer.validation.contracts import (
    ValidationRequirements,
)


_REQUIRED_METHODS = (
    "match_semantics",
    "match_structure",
    "analyze_evidence",
    "synthesize_candidates",
    "emit_representation_ir",
    "lower_for_target",
    "estimate_static_cost",
    "construction_requirements",
    "mount_requirements",
    "proof_or_error_contract",
    "benchmark_workloads",
    "validation_requirements",
)


@dataclass(frozen=True)
class ProviderRegistry:
    descriptors: RepresentationDescriptorRegistry
    providers: tuple[RepresentationProvider, ...] = ()

    @classmethod
    def from_providers(
        cls,
        *,
        descriptors: RepresentationDescriptorRegistry,
        providers: Iterable[RepresentationProvider],
    ) -> ProviderRegistry:
        registry = cls(descriptors=descriptors)
        for provider in providers:
            registry = registry.register(provider)
        return registry

    def register(
        self,
        provider: RepresentationProvider,
    ) -> ProviderRegistry:
        identity = getattr(provider, "identity", None)
        descriptor_id = getattr(provider, "descriptor_id", None)
        if not isinstance(identity, ProviderIdentity):
            raise ContractValidationError(
                "representation provider must declare ProviderIdentity"
            )
        if not isinstance(descriptor_id, str) or not descriptor_id:
            raise ContractValidationError(
                "representation provider must declare descriptor_id"
            )
        try:
            self.descriptors.get(descriptor_id)
        except KeyError as error:
            raise ContractValidationError(
                f"representation provider {identity.provider_id!r} references "
                f"unregistered descriptor {descriptor_id!r}"
            ) from error
        missing = [
            method
            for method in _REQUIRED_METHODS
            if not callable(getattr(provider, method, None))
        ]
        if missing:
            raise ContractValidationError(
                f"representation provider {identity.provider_id!r} does not "
                f"implement required methods {missing}"
            )
        if any(existing.identity == identity for existing in self.providers):
            raise ContractValidationError(
                f"representation provider identity {identity!r} is already registered"
            )
        return ProviderRegistry(
            descriptors=self.descriptors,
            providers=tuple(
                sorted(
                    (*self.providers, provider),
                    key=lambda item: item.identity,
                )
            ),
        )

    def run(self, problem: ProviderProblem) -> ProviderRegistryReport:
        evaluations = tuple(
            self._evaluate_provider(provider, problem) for provider in self.providers
        )
        unique, duplicates = _deduplicate_candidates(
            plan for evaluation in evaluations for plan in evaluation.candidates
        )
        return ProviderRegistryReport(
            evaluations=evaluations,
            candidates=unique,
            duplicate_candidates=duplicates,
        )

    def _evaluate_provider(
        self,
        provider: RepresentationProvider,
        problem: ProviderProblem,
    ) -> ProviderEvaluation:
        descriptor = self.descriptors.get(provider.descriptor_id)
        context = problem.bind_descriptor(descriptor)
        semantic: MatchAssessment | None = None
        structural: MatchAssessment | None = None
        evidence: EvidenceAssessment | None = None
        try:
            semantic = _match_assessment(
                provider.match_semantics(context),
                "semantic",
                context,
                evidence_required=False,
            )
            if not semantic.matched:
                return _evaluation(
                    provider,
                    status="declined",
                    semantic=semantic,
                )
            structural = _match_assessment(
                provider.match_structure(context),
                "structural",
                context,
                evidence_required=True,
            )
            if not structural.matched:
                return _evaluation(
                    provider,
                    status="declined",
                    semantic=semantic,
                    structural=structural,
                )
            evidence = _evidence_assessment(
                provider.analyze_evidence(context),
                context,
            )
            if not evidence.accepted:
                return _evaluation(
                    provider,
                    status="declined",
                    semantic=semantic,
                    structural=structural,
                    evidence=evidence,
                )
            if not set(structural.evidence_ids).issubset(evidence.evidence_ids):
                raise ContractValidationError(
                    "provider evidence analysis dropped structural match evidence"
                )
            raw_candidates = provider.synthesize_candidates(context, evidence)
            if not isinstance(raw_candidates, tuple) or not raw_candidates:
                raise ContractValidationError(
                    "matched provider must synthesize a non-empty candidate tuple"
                )
            candidates = tuple(
                _candidate_plan(
                    provider,
                    context,
                    evidence,
                    raw_candidate,
                )
                for raw_candidate in raw_candidates
            )
            if len({plan.candidate_id for plan in candidates}) != len(candidates):
                raise ContractValidationError(
                    "provider synthesized duplicate candidate identities"
                )
            return _evaluation(
                provider,
                status="completed",
                semantic=semantic,
                structural=structural,
                evidence=evidence,
                candidates=tuple(
                    sorted(candidates, key=lambda item: item.candidate_id)
                ),
            )
        except Exception as error:
            return _evaluation(
                provider,
                status="failed",
                semantic=semantic,
                structural=structural,
                evidence=evidence,
                error={
                    "type": type(error).__name__,
                    "message": str(error),
                },
            )


def _candidate_plan(
    provider: RepresentationProvider,
    context: ProviderContext,
    evidence: EvidenceAssessment,
    raw_candidate: Json,
) -> ProviderCandidatePlan:
    candidate = ContractDocument.from_json(
        raw_candidate,
        expected_schema=REPRESENTATION_CANDIDATE_SCHEMA,
    )
    document = candidate.to_json()
    if document["provider"] != provider.identity.to_json():
        raise ContractValidationError(
            "candidate provider identity does not match synthesizing provider"
        )
    if document["descriptor_id"] != provider.descriptor_id:
        raise ContractValidationError(
            "candidate descriptor does not match synthesizing provider"
        )
    _validate_candidate_source(document, context, evidence)
    representation_ir = RepresentationGraphDocument.from_json(
        provider.emit_representation_ir(context, candidate.to_json())
    )
    _validate_representation_ir_source(
        representation_ir.to_json(),
        document,
        context,
    )
    target_lowering = _provider_document(
        provider.lower_for_target(
            context,
            candidate.to_json(),
            representation_ir.to_json(),
        ),
        "target lowering",
    )
    estimate = provider.estimate_static_cost(
        context,
        candidate.to_json(),
        representation_ir.to_json(),
        deepcopy(target_lowering),
    )
    if not isinstance(estimate, StaticEstimate):
        raise ContractValidationError(
            "provider static cost estimator must return StaticEstimate"
        )
    construction = CandidateBuildPlan.from_json(
        provider.construction_requirements(context, candidate.to_json())
    )
    if construction.output_paths != tuple(
        artifact["path"] for artifact in document["artifact_declarations"]
    ):
        raise ContractValidationError(
            "candidate build-plan outputs must match candidate artifact declarations"
        )
    mount = RuntimeMountPlan.from_json(
        provider.mount_requirements(context, candidate.to_json()),
        candidate_id=str(document["candidate_id"]),
        build_plan=construction,
    )
    proof = _provider_document(
        provider.proof_or_error_contract(context, candidate.to_json()),
        "proof or error contract",
        schema_required=False,
    )
    if proof != document["behavioral_contract"]:
        raise ContractValidationError(
            "provider proof or error contract disagrees with candidate"
        )
    raw_workloads = provider.benchmark_workloads(context, candidate.to_json())
    if not isinstance(raw_workloads, tuple) or not raw_workloads:
        raise ContractValidationError(
            "provider must declare a non-empty benchmark workload tuple"
        )
    workloads = tuple(
        BenchmarkWorkload.from_json(workload)
        for workload in raw_workloads
    )
    validation = ValidationRequirements.from_json(
        provider.validation_requirements(context, candidate.to_json())
    )
    return ProviderCandidatePlan(
        provider=provider.identity,
        candidate=candidate,
        representation_ir=representation_ir,
        target_lowering=target_lowering,
        static_estimate=estimate,
        construction_requirements=construction,
        mount_requirements=mount,
        proof_or_error_contract=proof,
        benchmark_workloads=workloads,
        validation_requirements=validation,
    )


def _validate_candidate_source(
    candidate: Json,
    context: ProviderContext,
    evidence: EvidenceAssessment,
) -> None:
    expected = dict(
        zip(
            context.scope_ids,
            context.source_contract_digests,
            strict=True,
        )
    )
    scopes = candidate["scope_ids"]
    digests = candidate["source_contract_digests"]
    if scopes != sorted(scopes):
        raise ContractValidationError(
            "candidate scope_ids must use deterministic sorted order"
        )
    if any(
        expected.get(scope_id) != digest
        for scope_id, digest in zip(
            scopes,
            digests,
            strict=True,
        )
    ):
        raise ContractValidationError(
            "candidate does not preserve provider problem source contracts"
        )
    if set(candidate["evidence_refs"]) != set(evidence.evidence_ids):
        raise ContractValidationError(
            "candidate must retain the exact accepted evidence references"
        )


def _validate_representation_ir_source(
    representation_ir: Json,
    candidate: Json,
    context: ProviderContext,
) -> None:
    if representation_ir["candidate_id"] != candidate["candidate_id"]:
        raise ContractValidationError(
            "representation graph candidate_id does not match its candidate"
        )
    if representation_ir["scope_ids"] != candidate["scope_ids"]:
        raise ContractValidationError(
            "representation graph scopes do not match its candidate"
        )
    expected_digests = dict(
        zip(
            candidate["scope_ids"],
            candidate["source_contract_digests"],
            strict=True,
        )
    )
    if representation_ir["source_contract_digests"] != expected_digests:
        raise ContractValidationError(
            "representation graph source contracts do not match its candidate"
        )
    candidate_mode = candidate["behavioral_contract"]["mode"]
    representation_mode = representation_ir["confidence"]["mode"]
    if (candidate_mode == "exact") != (representation_mode == "exact"):
        raise ContractValidationError(
            "representation graph confidence disagrees with candidate behavior"
        )
    context_evidence = {evidence["evidence_id"] for evidence in context.evidence}
    cited_evidence = set(representation_ir["confidence"]["evidence_refs"])
    for collection in (
        "signals",
        "resources",
        "nodes",
        "absorbed_transforms",
        "physical_kernels",
        "unresolved",
        "correction_requests",
    ):
        for record in representation_ir[collection]:
            cited_evidence.update(record["provenance"]["evidence_refs"])
            cited_evidence.update(record.get("evidence_refs", []))
    if not cited_evidence <= context_evidence:
        raise ContractValidationError(
            "representation graph cites evidence outside the provider problem"
        )


def _match_assessment(
    value: object,
    kind: str,
    context: ProviderContext,
    *,
    evidence_required: bool,
) -> MatchAssessment:
    if not isinstance(value, MatchAssessment):
        raise ContractValidationError(
            f"provider {kind} matcher must return MatchAssessment"
        )
    _validate_evidence_ids(value.evidence_ids, context)
    if value.matched and evidence_required and not value.evidence_ids:
        raise ContractValidationError(
            f"matched provider {kind} assessment must cite evidence"
        )
    return value


def _evidence_assessment(
    value: object,
    context: ProviderContext,
) -> EvidenceAssessment:
    if not isinstance(value, EvidenceAssessment):
        raise ContractValidationError(
            "provider evidence analyzer must return EvidenceAssessment"
        )
    _validate_evidence_ids(value.evidence_ids, context)
    return value


def _validate_evidence_ids(
    evidence_ids: tuple[str, ...],
    context: ProviderContext,
) -> None:
    unknown = sorted(set(evidence_ids) - context.evidence_ids)
    if unknown:
        raise ContractValidationError(
            f"provider cited evidence outside its problem: {unknown}"
        )


def _provider_document(
    value: object,
    label: str,
    *,
    schema_required: bool = True,
) -> Json:
    if not isinstance(value, dict):
        raise ContractValidationError(f"provider {label} must be a JSON object")
    canonical_json_bytes(value)
    if schema_required and (
        not isinstance(value.get("schema"), str) or not value["schema"]
    ):
        raise ContractValidationError(f"provider {label} must declare a schema")
    return deepcopy(value)


def _evaluation(
    provider: RepresentationProvider,
    *,
    status: str,
    semantic: MatchAssessment | None = None,
    structural: MatchAssessment | None = None,
    evidence: EvidenceAssessment | None = None,
    candidates: tuple[ProviderCandidatePlan, ...] = (),
    error: Json | None = None,
) -> ProviderEvaluation:
    return ProviderEvaluation(
        provider=provider.identity,
        descriptor_id=provider.descriptor_id,
        status=status,
        semantic_match=semantic,
        structural_match=structural,
        evidence_assessment=evidence,
        candidates=candidates,
        error=error,
    )


def _deduplicate_candidates(
    plans: Iterable[ProviderCandidatePlan],
) -> tuple[tuple[ProviderCandidatePlan, ...], tuple[Json, ...]]:
    by_equivalence: dict[str, ProviderCandidatePlan] = {}
    duplicates = []
    for plan in sorted(plans, key=lambda item: item.candidate_id):
        document = plan.candidate.to_json()
        key = representation_candidate_equivalence_key(document)
        existing = by_equivalence.get(key)
        if existing is None:
            by_equivalence[key] = plan
            continue
        duplicates.append(
            {
                "equivalence_key": key,
                "kept_candidate_id": existing.candidate_id,
                "discarded_candidate_id": plan.candidate_id,
                "kept_provider": existing.provider.to_json(),
                "discarded_provider": plan.provider.to_json(),
            }
        )
    return (
        tuple(sorted(by_equivalence.values(), key=lambda item: item.candidate_id)),
        tuple(duplicates),
    )
