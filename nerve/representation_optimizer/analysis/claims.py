from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from nerve.compilation import Json
from nerve.representation_optimizer.analysis.context import ScopeAnalysisContext


@dataclass(frozen=True)
class AnalyzerResult:
    claims: tuple[Json, ...]
    details: Json

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "claims",
            tuple(json_value(item) for item in self.claims),
        )
        object.__setattr__(self, "details", json_value(self.details))


class StructuralAnalyzer(Protocol):
    analyzer_id: str
    version: str

    def analyze(self, context: ScopeAnalysisContext) -> AnalyzerResult: ...


def claim(
    *,
    kind: str,
    status: str,
    exact: bool,
    facts: Json,
) -> Json:
    if status not in {"supported", "rejected", "inconclusive"}:
        raise ValueError(f"invalid structural claim status {status!r}")
    return {
        "kind": kind,
        "status": status,
        "exact": exact,
        "facts": json_value(facts),
    }


def observation_facts(
    context: ScopeAnalysisContext,
    tensor_name: str,
) -> Json:
    return {
        "tensor": tensor_name,
        "observation": context.observation_domain(tensor_name),
    }


def array_digest(values: np.ndarray) -> str:
    normalized = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(normalized.dtype).encode())
    digest.update(repr(normalized.shape).encode())
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def tolerance_threshold(
    values: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> float:
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    return max(absolute_tolerance, relative_tolerance * scale)


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [json_value(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return [json_value(item) for item in value]
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return value
