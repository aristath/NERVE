from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nerve.representation_optimizer.analysis.context import ScopeAnalysisContext


@dataclass(frozen=True)
class ExactMatrixSvd:
    """One economy SVD reused by every analyzer of the same observation."""

    left_vectors: np.ndarray
    singular_values: np.ndarray
    right_vectors: np.ndarray


def exact_matrix_svd(
    context: ScopeAnalysisContext,
    tensor_name: str,
    matrix: np.ndarray,
) -> ExactMatrixSvd:
    observation = context.observation(tensor_name)
    key = (
        context.computations.observation_identity(observation),
        tuple(int(value) for value in matrix.shape),
        str(matrix.dtype),
    )

    def compute() -> ExactMatrixSvd:
        left, singular, right = np.linalg.svd(
            matrix,
            full_matrices=False,
        )
        return ExactMatrixSvd(
            left_vectors=np.ascontiguousarray(left),
            singular_values=np.ascontiguousarray(singular),
            right_vectors=np.ascontiguousarray(right),
        )

    return context.computations.get_or_compute(
        "exact_matrix_svd",
        key,
        compute,
    )
