from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nerve.compilation import Json
from nerve.representation_optimizer.analysis.tensor_repository import (
    TensorRepository,
)


@dataclass(frozen=True)
class AnalysisBudget:
    """Declared numerical-analysis limits, never implicit proof shortcuts."""

    exhaustive_element_limit: int | None = 1_000_000
    sampled_element_limit: int = 65_536
    decomposition_dimension_limit: int = 2_048
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 1e-5
    sparse_density_threshold: float = 0.1
    low_rank_ratio_threshold: float = 0.5
    codebook_ratio_threshold: float = 0.5

    def __post_init__(self) -> None:
        if (
            self.exhaustive_element_limit is not None
            and self.exhaustive_element_limit <= 0
        ):
            raise ValueError("exhaustive_element_limit must be positive or None")
        if self.sampled_element_limit <= 0:
            raise ValueError("sampled_element_limit must be positive")
        if self.decomposition_dimension_limit <= 0:
            raise ValueError("decomposition_dimension_limit must be positive")
        if self.absolute_tolerance < 0 or self.relative_tolerance < 0:
            raise ValueError("analysis tolerances must be non-negative")
        for name, value in (
            ("sparse_density_threshold", self.sparse_density_threshold),
            ("low_rank_ratio_threshold", self.low_rank_ratio_threshold),
            ("codebook_ratio_threshold", self.codebook_ratio_threshold),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")

    def to_json(self) -> Json:
        return {
            "exhaustive_element_limit": self.exhaustive_element_limit,
            "sampled_element_limit": self.sampled_element_limit,
            "decomposition_dimension_limit": self.decomposition_dimension_limit,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "sparse_density_threshold": self.sparse_density_threshold,
            "low_rank_ratio_threshold": self.low_rank_ratio_threshold,
            "codebook_ratio_threshold": self.codebook_ratio_threshold,
        }


@dataclass(frozen=True)
class ActivationTrace:
    """Reachable observations with an explicit, non-exhaustive domain."""

    domain: Json
    signals: dict[str, np.ndarray]
    trace_digest: str

    def __post_init__(self) -> None:
        if not self.domain:
            raise ValueError("activation trace must declare its observation domain")
        if not self.trace_digest:
            raise ValueError("activation trace must have a stable digest")
        for signal_id, values in self.signals.items():
            if not signal_id or not isinstance(values, np.ndarray):
                raise ValueError("activation trace signals must be named arrays")


@dataclass(frozen=True)
class ParameterBinding:
    binding_id: str
    component_id: str
    parameter_ref_id: str
    tensor_name: str
    semantic_role: str
    definition: Json


@dataclass
class ScopeAnalysisContext:
    package_id: str
    scope: Json
    source_contract: Json
    tensors: TensorRepository
    nodes: tuple[Json, ...]
    budget: AnalysisBudget
    activation_trace: ActivationTrace | None = None
    _observation_cache: dict[str, Any] = field(default_factory=dict)

    @property
    def scope_id(self) -> str:
        return str(self.scope["scope_id"])

    @property
    def source_contract_digest(self) -> str:
        return str(self.source_contract["contract_digest"])

    @property
    def parameters(self) -> tuple[ParameterBinding, ...]:
        bindings = []
        for parameter in self.scope["boundary"]["parameters"]:
            definition = parameter["definition"]
            tensor_name = definition.get("tensor")
            if not isinstance(tensor_name, str) or not tensor_name:
                continue
            bindings.append(
                ParameterBinding(
                    binding_id=str(parameter["id"]),
                    component_id=str(parameter["component_id"]),
                    parameter_ref_id=str(parameter["parameter_ref_id"]),
                    tensor_name=tensor_name,
                    semantic_role=str(definition.get("role", "parameter")),
                    definition=dict(definition),
                )
            )
        return tuple(bindings)

    def observation(self, tensor_name: str):
        cached = self._observation_cache.get(tensor_name)
        if cached is None:
            cached = self.tensors.observe(
                tensor_name,
                exhaustive_element_limit=self.budget.exhaustive_element_limit,
                sampled_element_limit=self.budget.sampled_element_limit,
            )
            self._observation_cache[tensor_name] = cached
        return cached

    def observation_domain(self, tensor_name: str) -> Json:
        observation = self.observation(tensor_name)
        return {
            "mode": "exhaustive" if observation.exhaustive else "deterministic_grid",
            "logical_shape": list(observation.logical_shape),
            "observed_shape": list(observation.values.shape),
            "logical_element_count": observation.logical_element_count,
            "observed_element_count": int(observation.values.size),
            "sample_indices": [list(indices) for indices in observation.sample_indices],
            "storage_dtype": observation.storage_dtype,
            "effective_values": observation.effective_values,
        }
