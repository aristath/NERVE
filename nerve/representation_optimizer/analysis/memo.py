from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import TypeVar, cast

import numpy as np

from nerve.representation_optimizer.analysis.tensor_repository import (
    TensorObservation,
)

T = TypeVar("T")


@dataclass
class AnalysisComputationMemo:
    """Run-local common-subexpression store for exact analysis computations."""

    _values: dict[tuple[str, Hashable], object] = field(default_factory=dict)
    _observation_identities: dict[int, tuple[TensorObservation, Hashable]] = (
        field(default_factory=dict)
    )
    _computations: Counter[str] = field(default_factory=Counter)
    _hits: Counter[str] = field(default_factory=Counter)

    def observation_identity(self, observation: TensorObservation) -> Hashable:
        object_id = id(observation)
        cached = self._observation_identities.get(object_id)
        if cached is not None and cached[0] is observation:
            return cached[1]
        values = np.ascontiguousarray(observation.values)
        digest = hashlib.sha256()
        digest.update(str(values.dtype).encode())
        digest.update(repr(values.shape).encode())
        digest.update(values.tobytes())
        identity: Hashable = (
            tuple(observation.logical_shape),
            observation.storage_dtype,
            observation.exhaustive,
            tuple(observation.sample_indices),
            observation.effective_values,
            str(values.dtype),
            tuple(values.shape),
            digest.hexdigest(),
        )
        self._observation_identities[object_id] = (observation, identity)
        return identity

    def get_or_compute(
        self,
        namespace: str,
        key: Hashable,
        compute: Callable[[], T],
    ) -> T:
        qualified = (namespace, key)
        if qualified in self._values:
            self._hits[namespace] += 1
            return cast(T, self._values[qualified])
        value = compute()
        _make_arrays_read_only(value)
        self._values[qualified] = value
        self._computations[namespace] += 1
        return value

    def statistics(self) -> dict[str, dict[str, int]]:
        namespaces = sorted(set(self._computations) | set(self._hits))
        return {
            namespace: {
                "computations": self._computations[namespace],
                "hits": self._hits[namespace],
            }
            for namespace in namespaces
        }


def _make_arrays_read_only(value: object) -> None:
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
        return
    if isinstance(value, dict):
        for item in value.values():
            _make_arrays_read_only(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _make_arrays_read_only(item)
        return
    fields = getattr(value, "__dataclass_fields__", None)
    if fields:
        for name in fields:
            _make_arrays_read_only(getattr(value, name))
