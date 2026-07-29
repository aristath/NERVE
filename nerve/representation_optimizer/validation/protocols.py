from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, ContextManager, Iterable, Protocol

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
class ValidationRoleMountRequest:
    plan_id: str
    candidate_id: str
    stage: str
    check: Json
    role: str
    implementation: Json
    matched_conditions: Json
    matched_conditions_digest: str
    seed: int
    block_index: int
    cancel_requested: Callable[[], bool] | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def to_json(self) -> Json:
        return {
            "plan_id": self.plan_id,
            "candidate_id": self.candidate_id,
            "stage": self.stage,
            "check": deepcopy(self.check),
            "role": self.role,
            "implementation": deepcopy(self.implementation),
            "matched_conditions": deepcopy(self.matched_conditions),
            "matched_conditions_digest": self.matched_conditions_digest,
            "seed": self.seed,
            "block_index": self.block_index,
        }


@dataclass(frozen=True)
class ValidationRoleExecutionRequest:
    plan_id: str
    candidate_id: str
    check: Json
    role: str
    implementation: Json
    matched_conditions: Json
    matched_conditions_digest: str
    seed: int
    reset_to_initial_state: bool = True

    def to_json(self) -> Json:
        return {
            "plan_id": self.plan_id,
            "candidate_id": self.candidate_id,
            "check": deepcopy(self.check),
            "role": self.role,
            "implementation": deepcopy(self.implementation),
            "matched_conditions": deepcopy(self.matched_conditions),
            "matched_conditions_digest": self.matched_conditions_digest,
            "seed": self.seed,
            "reset_to_initial_state": self.reset_to_initial_state,
        }


@dataclass(frozen=True)
class ValidationComparisonRequest:
    plan_id: str
    candidate_id: str
    check: Json
    seed: int
    behavioral_contract: Json

    def to_json(self) -> Json:
        return {
            "plan_id": self.plan_id,
            "candidate_id": self.candidate_id,
            "check": deepcopy(self.check),
            "seed": self.seed,
            "behavioral_contract": deepcopy(
                self.behavioral_contract
            ),
        }


class ValidationRoleExecutionSession(Protocol):
    """One role mounted through the ordinary runtime execution path."""

    @property
    def mount_event(self) -> Json:
        """Return validation_residency_event.v1 evidence for this mount."""

    def execute(self, request: ValidationRoleExecutionRequest) -> Json:
        """Execute one validation role and return its raw result."""

    def close(self) -> Json:
        """Release all residency and return validation_residency_event.v1."""


class BehavioralValidationAdapter(Protocol):
    """Backend-neutral access to ordinary runtime validation execution."""

    def validation_stage(
        self,
        stage: str,
        *,
        execution_scope: str,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> ContextManager[None]:
        """Own one scope's exclusive execution infrastructure for a stage."""

    def iter_fixture_artifact(
        self,
        relative_path: str,
        *,
        candidate_id: str,
        chunk_bytes: int = 8 * 1024 * 1024,
    ) -> Iterable[bytes]:
        """Read one immutable validation fixture or counterexample."""

    def open_session(
        self,
        request: ValidationRoleMountRequest,
    ) -> ValidationRoleExecutionSession:
        """Mount one role through the same public execution path as runtime."""

    def compare_results(
        self,
        request: ValidationComparisonRequest,
        reference_result: Json,
        candidate_result: Json,
    ) -> Json:
        """Compare two released role results without accelerator residency."""

    def iter_trace_artifact(
        self,
        relative_path: str,
        *,
        chunk_bytes: int = 8 * 1024 * 1024,
    ) -> Iterable[bytes]:
        """Read a raw validation trace without buffering the complete file."""
