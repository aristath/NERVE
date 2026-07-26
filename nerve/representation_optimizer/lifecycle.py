from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from nerve.compilation import Json
from nerve.representation_optimizer.contracts import (
    CONTRACT_DIGEST_SCHEMA,
    ContractValidationError,
    canonical_json_bytes,
    contract_digest,
    stable_contract_id,
)


CANDIDATE_LIFECYCLE_SCHEMA = "nerve.optimizer.candidate_lifecycle.v1"
OPTIMIZATION_SESSION_SCHEMA = "nerve.optimizer.session.v1"


class CandidateState(StrEnum):
    SYNTHESIZED = "synthesized"
    STAGED = "staged"
    STATICALLY_VALIDATED = "statically_validated"
    PREBENCHMARK_VALIDATED = "prebenchmark_validated"
    BENCHMARKED = "benchmarked"
    BEHAVIORALLY_VALIDATED = "behaviorally_validated"
    PROMOTABLE = "promotable"
    PUBLISHED = "published"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    FAILED = "failed"


TERMINAL_CANDIDATE_STATES = frozenset(
    {
        CandidateState.PUBLISHED,
        CandidateState.REJECTED,
        CandidateState.CANCELLED,
        CandidateState.FAILED,
    }
)
EVIDENCE_OPTIONAL_STATES = frozenset({CandidateState.CANCELLED})

_VALID_TRANSITIONS = {
    CandidateState.SYNTHESIZED: {
        CandidateState.STAGED,
        CandidateState.REJECTED,
        CandidateState.CANCELLED,
        CandidateState.FAILED,
    },
    CandidateState.STAGED: {
        CandidateState.STATICALLY_VALIDATED,
        CandidateState.REJECTED,
        CandidateState.CANCELLED,
        CandidateState.FAILED,
    },
    CandidateState.STATICALLY_VALIDATED: {
        CandidateState.PREBENCHMARK_VALIDATED,
        CandidateState.REJECTED,
        CandidateState.CANCELLED,
        CandidateState.FAILED,
    },
    CandidateState.PREBENCHMARK_VALIDATED: {
        CandidateState.BENCHMARKED,
        CandidateState.REJECTED,
        CandidateState.CANCELLED,
        CandidateState.FAILED,
    },
    CandidateState.BENCHMARKED: {
        CandidateState.BEHAVIORALLY_VALIDATED,
        CandidateState.REJECTED,
        CandidateState.CANCELLED,
        CandidateState.FAILED,
    },
    CandidateState.BEHAVIORALLY_VALIDATED: {
        CandidateState.PROMOTABLE,
        CandidateState.REJECTED,
        CandidateState.CANCELLED,
        CandidateState.FAILED,
    },
    CandidateState.PROMOTABLE: {
        CandidateState.PUBLISHED,
        CandidateState.REJECTED,
        CandidateState.CANCELLED,
        CandidateState.FAILED,
    },
}


@dataclass(frozen=True)
class CandidateLifecycle:
    candidate_id: str
    source_contract_digests: tuple[str, ...]
    state: CandidateState
    history: tuple[Json, ...]

    @classmethod
    def create(
        cls,
        candidate_id: str,
        source_contract_digests: tuple[str, ...],
    ) -> CandidateLifecycle:
        if not candidate_id:
            raise ContractValidationError("candidate lifecycle requires candidate_id")
        if not source_contract_digests:
            raise ContractValidationError(
                "candidate lifecycle requires source contract digests"
            )
        lifecycle = cls(
            candidate_id=candidate_id,
            source_contract_digests=source_contract_digests,
            state=CandidateState.SYNTHESIZED,
            history=(),
        )
        lifecycle._validate()
        return lifecycle

    @classmethod
    def from_json(cls, document: Json) -> CandidateLifecycle:
        _require_exact_fields(
            document,
            {
                "schema",
                "candidate_id",
                "source_contract_digests",
                "state",
                "history",
            },
            "candidate lifecycle",
        )
        if document["schema"] != CANDIDATE_LIFECYCLE_SCHEMA:
            raise ContractValidationError(
                f"unsupported candidate lifecycle schema {document['schema']!r}"
            )
        try:
            state = CandidateState(document["state"])
        except (TypeError, ValueError) as error:
            raise ContractValidationError(
                f"invalid candidate lifecycle state {document['state']!r}"
            ) from error
        source_digests = document["source_contract_digests"]
        history = document["history"]
        if not isinstance(source_digests, list) or not all(
            isinstance(item, str) for item in source_digests
        ):
            raise ContractValidationError(
                "candidate lifecycle source_contract_digests must be strings"
            )
        if not isinstance(history, list) or not all(
            isinstance(item, dict) for item in history
        ):
            raise ContractValidationError("candidate lifecycle history must be objects")
        lifecycle = cls(
            candidate_id=str(document["candidate_id"]),
            source_contract_digests=tuple(source_digests),
            state=state,
            history=tuple(deepcopy(history)),
        )
        lifecycle._validate()
        return lifecycle

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_CANDIDATE_STATES

    def transition(
        self,
        next_state: CandidateState,
        *,
        evidence_refs: tuple[str, ...],
        reason: str,
    ) -> CandidateLifecycle:
        allowed = _VALID_TRANSITIONS.get(self.state, set())
        if next_state not in allowed:
            raise ContractValidationError(
                f"candidate cannot transition from {self.state.value!r} "
                f"to {next_state.value!r}"
            )
        if not reason:
            raise ContractValidationError("candidate transition requires a reason")
        if next_state not in EVIDENCE_OPTIONAL_STATES and not evidence_refs:
            raise ContractValidationError(
                f"candidate transition to {next_state.value!r} requires evidence"
            )
        event = {
            "sequence": len(self.history),
            "from": self.state.value,
            "to": next_state.value,
            "evidence_refs": list(evidence_refs),
            "reason": reason,
        }
        return CandidateLifecycle(
            candidate_id=self.candidate_id,
            source_contract_digests=self.source_contract_digests,
            state=next_state,
            history=(*self.history, event),
        )

    def to_json(self) -> Json:
        document = {
            "schema": CANDIDATE_LIFECYCLE_SCHEMA,
            "candidate_id": self.candidate_id,
            "source_contract_digests": list(self.source_contract_digests),
            "state": self.state.value,
            "history": deepcopy(list(self.history)),
        }
        canonical_json_bytes(document)
        return document

    def _validate(self) -> None:
        if not self.candidate_id:
            raise ContractValidationError("candidate lifecycle requires candidate_id")
        if not self.source_contract_digests or len(self.source_contract_digests) != len(
            set(self.source_contract_digests)
        ):
            raise ContractValidationError(
                "candidate lifecycle source digests must be non-empty and unique"
            )
        if any(not _is_contract_digest(digest) for digest in self.source_contract_digests):
            raise ContractValidationError(
                "candidate lifecycle contains an invalid source contract digest"
            )
        current = CandidateState.SYNTHESIZED
        for sequence, event in enumerate(self.history):
            _require_exact_fields(
                event,
                {"sequence", "from", "to", "evidence_refs", "reason"},
                f"candidate lifecycle history[{sequence}]",
            )
            if event["sequence"] != sequence:
                raise ContractValidationError(
                    "candidate lifecycle history sequence is not contiguous"
                )
            if event["from"] != current.value:
                raise ContractValidationError(
                    "candidate lifecycle history does not form a continuous chain"
                )
            try:
                next_state = CandidateState(event["to"])
            except (TypeError, ValueError) as error:
                raise ContractValidationError(
                    f"candidate lifecycle history has invalid state {event['to']!r}"
                ) from error
            if next_state not in _VALID_TRANSITIONS.get(current, set()):
                raise ContractValidationError(
                    f"candidate lifecycle contains invalid transition "
                    f"{current.value!r} -> {next_state.value!r}"
                )
            if not isinstance(event["evidence_refs"], list) or not all(
                isinstance(item, str) and item for item in event["evidence_refs"]
            ):
                raise ContractValidationError(
                    "candidate lifecycle evidence_refs must be non-empty strings"
                )
            if next_state not in EVIDENCE_OPTIONAL_STATES and not event["evidence_refs"]:
                raise ContractValidationError(
                    f"candidate transition to {next_state.value!r} requires evidence"
                )
            if not isinstance(event["reason"], str) or not event["reason"]:
                raise ContractValidationError(
                    "candidate lifecycle transition reason must be non-empty"
                )
            current = next_state
        if current != self.state:
            raise ContractValidationError(
                "candidate lifecycle state does not match its transition history"
            )


@dataclass(frozen=True)
class OptimizationSession:
    session_id: str
    package_id: str
    exact_baseline_digest: str
    candidates: tuple[CandidateLifecycle, ...] = ()

    @classmethod
    def create(
        cls,
        package_id: str,
        exact_baseline_digest: str,
    ) -> OptimizationSession:
        if not package_id or not exact_baseline_digest:
            raise ContractValidationError(
                "optimization session requires package and exact baseline identities"
            )
        if not _is_contract_digest(exact_baseline_digest):
            raise ContractValidationError(
                "optimization session exact baseline digest is invalid"
            )
        return cls(
            session_id=stable_contract_id(
                "optimization_session",
                package_id,
                exact_baseline_digest,
            ),
            package_id=package_id,
            exact_baseline_digest=exact_baseline_digest,
        )

    @classmethod
    def from_json(cls, document: Json) -> OptimizationSession:
        _require_exact_fields(
            document,
            {
                "schema",
                "session_id",
                "package_id",
                "exact_baseline_digest",
                "candidates",
            },
            "optimization session",
        )
        if document["schema"] != OPTIMIZATION_SESSION_SCHEMA:
            raise ContractValidationError(
                f"unsupported optimization session schema {document['schema']!r}"
            )
        candidates = document["candidates"]
        if not isinstance(candidates, list):
            raise ContractValidationError(
                "optimization session candidates must be a list"
            )
        session = cls(
            session_id=str(document["session_id"]),
            package_id=str(document["package_id"]),
            exact_baseline_digest=str(document["exact_baseline_digest"]),
            candidates=tuple(
                CandidateLifecycle.from_json(candidate) for candidate in candidates
            ),
        )
        session._validate()
        return session

    def register_candidate(
        self,
        candidate_id: str,
        source_contract_digests: tuple[str, ...],
    ) -> OptimizationSession:
        if any(candidate.candidate_id == candidate_id for candidate in self.candidates):
            raise ContractValidationError(
                f"candidate {candidate_id!r} is already registered"
            )
        candidate = CandidateLifecycle.create(candidate_id, source_contract_digests)
        return OptimizationSession(
            session_id=self.session_id,
            package_id=self.package_id,
            exact_baseline_digest=self.exact_baseline_digest,
            candidates=(*self.candidates, candidate),
        )

    def transition_candidate(
        self,
        candidate_id: str,
        next_state: CandidateState,
        *,
        evidence_refs: tuple[str, ...],
        reason: str,
    ) -> OptimizationSession:
        found = False
        candidates = []
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                found = True
                candidate = candidate.transition(
                    next_state,
                    evidence_refs=evidence_refs,
                    reason=reason,
                )
            candidates.append(candidate)
        if not found:
            raise ContractValidationError(
                f"candidate {candidate_id!r} is not registered"
            )
        return OptimizationSession(
            session_id=self.session_id,
            package_id=self.package_id,
            exact_baseline_digest=self.exact_baseline_digest,
            candidates=tuple(candidates),
        )

    def to_json(self) -> Json:
        document = {
            "schema": OPTIMIZATION_SESSION_SCHEMA,
            "session_id": self.session_id,
            "package_id": self.package_id,
            "exact_baseline_digest": self.exact_baseline_digest,
            "candidates": [candidate.to_json() for candidate in self.candidates],
        }
        canonical_json_bytes(document)
        return document

    @property
    def digest(self) -> str:
        return contract_digest(self.to_json())

    def _validate(self) -> None:
        expected_id = stable_contract_id(
            "optimization_session",
            self.package_id,
            self.exact_baseline_digest,
        )
        if self.session_id != expected_id:
            raise ContractValidationError(
                f"optimization session id must be {expected_id!r}"
            )
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ContractValidationError(
                "optimization session contains duplicate candidate ids"
            )


def _require_exact_fields(document: Json, fields: set[str], path: str) -> None:
    if not isinstance(document, dict):
        raise ContractValidationError(f"{path} must be an object")
    actual = set(document)
    missing = sorted(fields - actual)
    unknown = sorted(actual - fields)
    if missing:
        raise ContractValidationError(f"{path} is missing fields {missing}")
    if unknown:
        raise ContractValidationError(f"{path} has unknown fields {unknown}")


def _is_contract_digest(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    prefix = f"{CONTRACT_DIGEST_SCHEMA}:"
    hexadecimal = value.removeprefix(prefix)
    return (
        value.startswith(prefix)
        and len(hexadecimal) == 64
        and all(character in "0123456789abcdef" for character in hexadecimal)
    )
