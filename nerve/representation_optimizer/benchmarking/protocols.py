from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, Protocol

from nerve.compilation import Json


@dataclass(frozen=True)
class BenchmarkMountRequest:
    plan_id: str
    role: str
    implementation: Json
    workload: Json
    matched_conditions: Json
    matched_conditions_digest: str
    seed: int
    block_index: int

    def to_json(self) -> Json:
        return {
            "plan_id": self.plan_id,
            "role": self.role,
            "implementation": deepcopy(self.implementation),
            "workload": deepcopy(self.workload),
            "matched_conditions": deepcopy(self.matched_conditions),
            "matched_conditions_digest": self.matched_conditions_digest,
            "seed": self.seed,
            "block_index": self.block_index,
        }


@dataclass(frozen=True)
class BenchmarkExecutionRequest:
    plan_id: str
    role: str
    implementation_id: str
    workload: Json
    matched_conditions: Json
    matched_conditions_digest: str
    phase: str
    seed: int
    pair_index: int
    order_index: int
    reset_to_initial_state: bool = True

    def to_json(self) -> Json:
        return {
            "plan_id": self.plan_id,
            "role": self.role,
            "implementation_id": self.implementation_id,
            "workload": deepcopy(self.workload),
            "matched_conditions": deepcopy(self.matched_conditions),
            "matched_conditions_digest": self.matched_conditions_digest,
            "phase": self.phase,
            "seed": self.seed,
            "pair_index": self.pair_index,
            "order_index": self.order_index,
            "reset_to_initial_state": self.reset_to_initial_state,
        }


class NormalExecutionSession(Protocol):
    """One mount of the same public execution implementation used at runtime."""

    @property
    def mount_event(self) -> Json:
        """Return benchmark_residency_event.v1 evidence for this mount."""

    def execute(self, request: BenchmarkExecutionRequest) -> Json:
        """Execute through the implementation's ordinary public run method."""

    def close(self) -> Json:
        """Release all residency and return benchmark_residency_event.v1."""


class NormalExecutionAdapter(Protocol):
    """Mount reference and candidate implementations through one normal API."""

    def iter_fixture_artifact(
        self,
        relative_path: str,
        *,
        candidate_id: str,
        chunk_bytes: int = 8 * 1024 * 1024,
    ) -> Iterable[bytes]:
        """Read one immutable benchmark input or initial-state artifact."""

    def open_session(
        self,
        request: BenchmarkMountRequest,
    ) -> NormalExecutionSession:
        """Mount one implementation under the supplied matched conditions."""

    def iter_trace_artifact(
        self,
        relative_path: str,
        *,
        chunk_bytes: int = 8 * 1024 * 1024,
    ) -> Iterable[bytes]:
        """Read a raw trace emitted by normal execution without buffering it."""
