from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Iterable

from nerve.compilation import Json, check_compile_cancelled
from nerve.representation_optimizer.contracts import (
    ALGEBRAIC_EVIDENCE_SCHEMA,
    HARDWARE_PROCESS_PROFILE_SCHEMA,
    OPTIMIZATION_SCOPE_SCHEMA,
    REPRESENTATION_DESCRIPTOR_SCHEMA,
    SOURCE_BEHAVIOR_CONTRACT_SCHEMA,
    ContractDocument,
    ContractValidationError,
    canonical_json_bytes,
)
from nerve.representation_optimizer.benchmarking.contracts import (
    BenchmarkWorkload,
)
from nerve.representation_optimizer.mounting import RuntimeMountPlan
from nerve.representation_optimizer.representation_ir.contracts import (
    RepresentationGraphDocument,
)
from nerve.representation_optimizer.staging.contracts import CandidateBuildPlan
from nerve.representation_optimizer.validation.contracts import (
    ValidationRequirements,
)
from nerve.representation_optimizer.providers.source_artifacts import (
    SourceArtifactResolver,
)


@dataclass(frozen=True, order=True)
class ProviderIdentity:
    provider_id: str
    version: str

    def __post_init__(self) -> None:
        if not self.provider_id or not self.version:
            raise ContractValidationError(
                "representation provider identity requires id and version"
            )

    def to_json(self) -> Json:
        return {"id": self.provider_id, "version": self.version}


@dataclass(frozen=True)
class MatchAssessment:
    matched: bool
    reasons: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty_unique_strings(self.reasons, "provider match reasons")
        _unique_strings(self.evidence_ids, "provider match evidence_ids")


@dataclass(frozen=True)
class EvidenceAssessment:
    accepted: bool
    evidence_ids: tuple[str, ...]
    facts: Json
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty_unique_strings(self.reasons, "provider evidence reasons")
        _unique_strings(self.evidence_ids, "provider evidence ids")
        if self.accepted and not self.evidence_ids:
            raise ContractValidationError(
                "accepted provider evidence must cite source evidence"
            )
        _json_object(self.facts, "provider evidence facts")


@dataclass(frozen=True)
class StaticEstimate:
    feasible: bool
    permanent_bytes: int
    transient_bytes: int
    construction_nanoseconds: int | None
    steady_state_work: Json
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("permanent_bytes", self.permanent_bytes),
            ("transient_bytes", self.transient_bytes),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ContractValidationError(
                    f"provider static estimate {name} must be non-negative"
                )
        if self.construction_nanoseconds is not None and (
            not isinstance(self.construction_nanoseconds, int)
            or isinstance(self.construction_nanoseconds, bool)
            or self.construction_nanoseconds < 0
        ):
            raise ContractValidationError(
                "provider static estimate construction_nanoseconds must be "
                "non-negative or null"
            )
        _json_object(
            self.steady_state_work,
            "provider static estimate steady_state_work",
        )
        _nonempty_unique_strings(
            self.reasons,
            "provider static estimate reasons",
        )

    def to_json(self) -> Json:
        return {
            "feasible": self.feasible,
            "permanent_bytes": self.permanent_bytes,
            "transient_bytes": self.transient_bytes,
            "construction_nanoseconds": self.construction_nanoseconds,
            "steady_state_work": deepcopy(self.steady_state_work),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ProviderContext:
    package_id: str
    _scopes: tuple[ContractDocument, ...]
    _source_contracts: tuple[ContractDocument, ...]
    _evidence: tuple[ContractDocument, ...]
    _hardware_profile: ContractDocument
    _descriptor: ContractDocument
    _source_artifacts: SourceArtifactResolver | None
    _shared_cache: dict[str, object]
    _cancel_requested: Callable[[], bool] | None = None

    @property
    def scopes(self) -> tuple[Json, ...]:
        return tuple(document.to_json() for document in self._scopes)

    @property
    def source_contracts(self) -> tuple[Json, ...]:
        return tuple(document.to_json() for document in self._source_contracts)

    @property
    def evidence(self) -> tuple[Json, ...]:
        return tuple(document.to_json() for document in self._evidence)

    @property
    def hardware_profile(self) -> Json:
        return self._hardware_profile.to_json()

    @property
    def descriptor(self) -> Json:
        return self._descriptor.to_json()

    @property
    def scope_ids(self) -> tuple[str, ...]:
        return tuple(str(document.to_json()["scope_id"]) for document in self._scopes)

    @property
    def source_contract_digests(self) -> tuple[str, ...]:
        return tuple(
            str(document.to_json()["contract_digest"])
            for document in self._source_contracts
        )

    @property
    def evidence_ids(self) -> frozenset[str]:
        return frozenset(
            str(document.to_json()["evidence_id"]) for document in self._evidence
        )

    @property
    def source_artifacts(self) -> SourceArtifactResolver:
        self.checkpoint()
        if self._source_artifacts is None:
            raise ContractValidationError(
                "representation provider requires source artifact access, but "
                "this optimization problem has no source artifact resolver"
            )
        return self._source_artifacts

    def checkpoint(self) -> None:
        """Stop provider work promptly without weakening phase isolation."""

        check_compile_cancelled(self._cancel_requested)

    def restrict_to_scope(self, scope_id: str) -> ProviderContext:
        """Return this provider problem restricted to one semantic scope."""

        matches = [
            context
            for context in self.single_scope_contexts()
            if context.scope_ids == (scope_id,)
        ]
        if len(matches) != 1:
            raise ContractValidationError(
                f"provider context has no unique scope {scope_id!r}"
            )
        return matches[0]

    def single_scope_contexts(self) -> tuple[ProviderContext, ...]:
        """Partition the problem in one linear pass over its evidence."""

        cache_key = (
            "provider_context.single_scope_contexts:"
            f"{self._descriptor.digest}"
        )
        cached = self._shared_cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        evidence_by_scope: dict[str, list[ContractDocument]] = {
            scope_id: [] for scope_id in self.scope_ids
        }
        for evidence in self._evidence:
            scope_id = str(evidence.to_json()["scope_id"])
            evidence_by_scope[scope_id].append(evidence)
        contexts = tuple(
            ProviderContext(
                package_id=self.package_id,
                _scopes=(scope,),
                _source_contracts=(contract,),
                _evidence=tuple(evidence_by_scope[scope_id]),
                _hardware_profile=self._hardware_profile,
                _descriptor=self._descriptor,
                _source_artifacts=self._source_artifacts,
                _shared_cache=self._shared_cache,
                _cancel_requested=self._cancel_requested,
            )
            for scope, contract, scope_id in (
                (
                    scope,
                    contract,
                    str(scope.to_json()["scope_id"]),
                )
                for scope, contract in zip(
                    self._scopes,
                    self._source_contracts,
                    strict=True,
                )
            )
        )
        self._shared_cache[cache_key] = contexts
        return contexts

    def memoized(self, key: str, factory: Callable[[], object]) -> object:
        cached = self._shared_cache.get(key)
        if cached is None:
            cached = factory()
            self._shared_cache[key] = cached
        return cached


@dataclass(frozen=True)
class ProviderProblem:
    package_id: str
    _scopes: tuple[ContractDocument, ...]
    _source_contracts: tuple[ContractDocument, ...]
    _evidence: tuple[ContractDocument, ...]
    _hardware_profile: ContractDocument
    _source_artifacts: SourceArtifactResolver | None
    _shared_cache: dict[str, object]

    @classmethod
    def from_documents(
        cls,
        *,
        package_id: str,
        scopes: Iterable[Json | ContractDocument],
        source_contracts: Iterable[Json | ContractDocument],
        evidence: Iterable[Json | ContractDocument],
        hardware_profile: Json | ContractDocument,
        source_artifacts: SourceArtifactResolver | None = None,
    ) -> ProviderProblem:
        if not package_id:
            raise ContractValidationError("provider problem requires package_id")
        parsed_scopes = _documents(scopes, OPTIMIZATION_SCOPE_SCHEMA)
        parsed_contracts = _documents(
            source_contracts,
            SOURCE_BEHAVIOR_CONTRACT_SCHEMA,
        )
        parsed_evidence = _documents(evidence, ALGEBRAIC_EVIDENCE_SCHEMA)
        parsed_profile = _document(
            hardware_profile,
            HARDWARE_PROCESS_PROFILE_SCHEMA,
        )
        scope_json = [item.to_json() for item in parsed_scopes]
        contract_json = [item.to_json() for item in parsed_contracts]
        scope_ids = [str(item["scope_id"]) for item in scope_json]
        if not scope_ids or len(scope_ids) != len(set(scope_ids)):
            raise ContractValidationError(
                "provider problem scopes must be non-empty and unique"
            )
        if any(item["package_id"] != package_id for item in scope_json):
            raise ContractValidationError(
                "provider problem scope package does not match package_id"
            )
        contracts = {str(item["scope_id"]): item for item in contract_json}
        if set(contracts) != set(scope_ids) or len(contracts) != len(contract_json):
            raise ContractValidationError(
                "provider problem requires exactly one source contract per scope"
            )
        for scope in scope_json:
            contract = contracts[str(scope["scope_id"])]
            if scope["source_contract_digest"] != contract["contract_digest"]:
                raise ContractValidationError(
                    "provider scope and source contract digests disagree"
                )
        evidence_ids = set()
        for item in (document.to_json() for document in parsed_evidence):
            if item["evidence_id"] in evidence_ids:
                raise ContractValidationError(
                    "provider problem contains duplicate evidence"
                )
            evidence_ids.add(item["evidence_id"])
            if item["scope_id"] not in contracts:
                raise ContractValidationError(
                    "provider evidence belongs to an unrelated scope"
                )
            if (
                item["source_contract_digest"]
                != contracts[item["scope_id"]]["contract_digest"]
            ):
                raise ContractValidationError(
                    "provider evidence source contract digest disagrees"
                )
        scope_documents = {
            str(item.to_json()["scope_id"]): item for item in parsed_scopes
        }
        contract_documents = {
            str(item.to_json()["scope_id"]): item for item in parsed_contracts
        }
        ordered_scope_ids = sorted(scope_ids)
        return cls(
            package_id=package_id,
            _scopes=tuple(scope_documents[scope_id] for scope_id in ordered_scope_ids),
            _source_contracts=tuple(
                contract_documents[scope_id] for scope_id in ordered_scope_ids
            ),
            _evidence=tuple(
                sorted(
                    parsed_evidence,
                    key=lambda item: str(item.to_json()["evidence_id"]),
                )
            ),
            _hardware_profile=parsed_profile,
            _source_artifacts=source_artifacts,
            _shared_cache={},
        )

    def bind_descriptor(
        self,
        descriptor: ContractDocument,
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> ProviderContext:
        if descriptor.schema != REPRESENTATION_DESCRIPTOR_SCHEMA:
            raise ContractValidationError(
                "provider context requires a representation descriptor"
            )
        return ProviderContext(
            package_id=self.package_id,
            _scopes=self._scopes,
            _source_contracts=self._source_contracts,
            _evidence=self._evidence,
            _hardware_profile=self._hardware_profile,
            _descriptor=descriptor,
            _source_artifacts=self._source_artifacts,
            _shared_cache=self._shared_cache,
            _cancel_requested=cancel_requested,
        )


@dataclass(frozen=True)
class ProviderCandidatePlan:
    provider: ProviderIdentity
    candidate: ContractDocument
    representation_ir: RepresentationGraphDocument
    target_lowering: Json
    static_estimate: StaticEstimate
    construction_requirements: CandidateBuildPlan
    mount_requirements: RuntimeMountPlan
    proof_or_error_contract: Json
    benchmark_workloads: tuple[BenchmarkWorkload, ...]
    validation_requirements: ValidationRequirements

    @property
    def candidate_id(self) -> str:
        return str(self.candidate.to_json()["candidate_id"])


@dataclass(frozen=True)
class ProviderEvaluation:
    provider: ProviderIdentity
    descriptor_id: str
    status: str
    semantic_match: MatchAssessment | None
    structural_match: MatchAssessment | None
    evidence_assessment: EvidenceAssessment | None
    candidates: tuple[ProviderCandidatePlan, ...]
    error: Json | None = None


@dataclass(frozen=True)
class ProviderRegistryReport:
    evaluations: tuple[ProviderEvaluation, ...]
    candidates: tuple[ProviderCandidatePlan, ...]
    duplicate_candidates: tuple[Json, ...]


def _documents(
    values: Iterable[Json | ContractDocument],
    schema: str,
) -> tuple[ContractDocument, ...]:
    return tuple(_document(value, schema) for value in values)


def _document(
    value: Json | ContractDocument,
    schema: str,
) -> ContractDocument:
    if isinstance(value, ContractDocument):
        if value.schema != schema:
            raise ContractValidationError(
                f"provider document must use schema {schema!r}"
            )
        return ContractDocument.from_json(
            value.to_json(),
            expected_schema=schema,
        )
    return ContractDocument.from_json(value, expected_schema=schema)


def _json_object(value: object, label: str) -> Json:
    if not isinstance(value, dict):
        raise ContractValidationError(f"{label} must be a JSON object")
    canonical_json_bytes(value)
    return deepcopy(value)


def _unique_strings(values: tuple[str, ...], label: str) -> None:
    if not all(isinstance(value, str) and value for value in values):
        raise ContractValidationError(f"{label} must contain non-empty strings")
    if len(values) != len(set(values)):
        raise ContractValidationError(f"{label} must be unique")
    if values != tuple(sorted(values)):
        raise ContractValidationError(f"{label} must be sorted")


def _nonempty_unique_strings(values: tuple[str, ...], label: str) -> None:
    _unique_strings(values, label)
    if not values:
        raise ContractValidationError(f"{label} must not be empty")
