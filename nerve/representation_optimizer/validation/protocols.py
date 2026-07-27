from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, Protocol

from nerve.compilation import Json


@dataclass(frozen=True)
class ProofRequest:
    plan_id: str
    candidate_id: str
    obligation: str
    verifier_id: str
    source_contract_digests: tuple[str, ...]
    construction_record_digest: str
    reference_implementation: Json
    candidate_implementation: Json

    def to_json(self) -> Json:
        return {
            "plan_id": self.plan_id,
            "candidate_id": self.candidate_id,
            "obligation": self.obligation,
            "verifier_id": self.verifier_id,
            "source_contract_digests": list(
                self.source_contract_digests
            ),
            "construction_record_digest": self.construction_record_digest,
            "reference_implementation": deepcopy(
                self.reference_implementation
            ),
            "candidate_implementation": deepcopy(
                self.candidate_implementation
            ),
        }


class ExactProofVerifier(Protocol):
    """A deterministic verifier for one named algebraic obligation."""

    @property
    def verifier_id(self) -> str:
        """Stable implementation identity used by validation requirements."""

    def verify(self, request: ProofRequest) -> Json:
        """Return a proof_result.v1 document bound to the request."""

    def iter_proof_artifact(
        self,
        relative_path: str,
        *,
        chunk_bytes: int = 8 * 1024 * 1024,
    ) -> Iterable[bytes]:
        """Stream a certificate referenced by the verifier's proof result."""


@dataclass(frozen=True)
class ValidationStageMountRequest:
    plan_id: str
    stage: str
    implementations: Json
    matched_conditions: Json
    matched_conditions_digest: str

    def to_json(self) -> Json:
        return {
            "plan_id": self.plan_id,
            "stage": self.stage,
            "implementations": deepcopy(self.implementations),
            "matched_conditions": deepcopy(self.matched_conditions),
            "matched_conditions_digest": self.matched_conditions_digest,
        }


@dataclass(frozen=True)
class ValidationExecutionRequest:
    plan_id: str
    check: Json
    reference_implementation: Json
    candidate_implementation: Json
    matched_conditions: Json
    matched_conditions_digest: str
    seed: int
    reset_to_initial_state: bool = True

    def to_json(self) -> Json:
        return {
            "plan_id": self.plan_id,
            "check": deepcopy(self.check),
            "reference_implementation": deepcopy(
                self.reference_implementation
            ),
            "candidate_implementation": deepcopy(
                self.candidate_implementation
            ),
            "matched_conditions": deepcopy(self.matched_conditions),
            "matched_conditions_digest": self.matched_conditions_digest,
            "seed": self.seed,
            "reset_to_initial_state": self.reset_to_initial_state,
        }


class ValidationExecutionSession(Protocol):
    """Normal runtime execution mounted for one validation stage."""

    @property
    def mount_event(self) -> Json:
        """Return validation_residency_event.v1 evidence for this mount."""

    def execute_pair(self, request: ValidationExecutionRequest) -> Json:
        """Compare exact and candidate implementations via normal execution."""

    def close(self) -> Json:
        """Release all residency and return validation_residency_event.v1."""


class BehavioralValidationAdapter(Protocol):
    """Backend-neutral access to ordinary runtime validation execution."""

    def iter_fixture_artifact(
        self,
        relative_path: str,
        *,
        candidate_id: str,
        chunk_bytes: int = 8 * 1024 * 1024,
    ) -> Iterable[bytes]:
        """Read one immutable validation fixture or counterexample."""

    def open_stage(
        self,
        request: ValidationStageMountRequest,
    ) -> ValidationExecutionSession:
        """Mount a stage through the same public execution path as runtime."""

    def iter_trace_artifact(
        self,
        relative_path: str,
        *,
        chunk_bytes: int = 8 * 1024 * 1024,
    ) -> Iterable[bytes]:
        """Read a raw validation trace without buffering the complete file."""
