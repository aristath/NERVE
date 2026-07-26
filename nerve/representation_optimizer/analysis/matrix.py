from __future__ import annotations

import math

import numpy as np

from nerve.representation_optimizer.analysis.claims import (
    AnalyzerResult,
    claim,
    observation_facts,
    tolerance_threshold,
)
from nerve.representation_optimizer.analysis.context import ScopeAnalysisContext


class MatrixStructureAnalyzer:
    analyzer_id = "matrix_and_tensor_structure"
    version = "1"

    def analyze(self, context: ScopeAnalysisContext) -> AnalyzerResult:
        claims = []
        details = []
        for parameter in context.parameters:
            observation = context.observation(parameter.tensor_name)
            values = observation.values.astype(np.float64, copy=False)
            base = {
                **observation_facts(context, parameter.tensor_name),
                "semantic_role": parameter.semantic_role,
            }
            threshold = tolerance_threshold(
                values,
                absolute_tolerance=context.budget.absolute_tolerance,
                relative_tolerance=context.budget.relative_tolerance,
            )
            if values.ndim == 1:
                vector_claims, vector_details = _analyze_vector(
                    values,
                    base,
                    threshold,
                    observation.exhaustive,
                )
                claims.extend(vector_claims)
                details.append(vector_details)
                continue
            if values.ndim < 2:
                continue

            matrix = values.reshape(-1, values.shape[-1])
            if max(matrix.shape) > context.budget.decomposition_dimension_limit:
                claims.extend(
                    _inconclusive_decomposition_claims(
                        base,
                        matrix.shape,
                        context.budget.decomposition_dimension_limit,
                    )
                )
                topology_claims, topology_details = _analyze_tensor_topology(
                    context,
                    values,
                    base,
                    threshold,
                    observation.exhaustive,
                )
                claims.extend(topology_claims)
                details.append(topology_details)
                continue

            matrix_claims, matrix_details = _analyze_matrix(
                matrix,
                base=base,
                threshold=threshold,
                exhaustive=observation.exhaustive,
                low_rank_ratio=context.budget.low_rank_ratio_threshold,
                sparse_density=context.budget.sparse_density_threshold,
            )
            claims.extend(matrix_claims)
            topology_claims, topology_details = _analyze_tensor_topology(
                context,
                values,
                base,
                threshold,
                observation.exhaustive,
            )
            claims.extend(topology_claims)
            details.append({**matrix_details, **topology_details})
        return AnalyzerResult(claims=tuple(claims), details={"tensors": details})


def _analyze_vector(
    values: np.ndarray,
    base: dict,
    threshold: float,
    exhaustive: bool,
) -> tuple[list[dict], dict]:
    norm = float(np.linalg.norm(values))
    mean = float(np.mean(values)) if values.size else 0.0
    centered = values - mean
    invariant = bool(np.max(np.abs(centered)) <= threshold) if values.size else True
    claims = [
        claim(
            kind="normalization_invariant",
            status="supported" if invariant else "rejected",
            exact=exhaustive and threshold == 0,
            facts={
                **base,
                "vector_norm": norm,
                "mean": mean,
                "maximum_centered_deviation": (
                    float(np.max(np.abs(centered))) if values.size else 0.0
                ),
                "tolerance": threshold,
            },
        )
    ]
    spectral = _spectral_facts(values)
    claims.append(
        claim(
            kind="spectral_structure",
            status=(
                "supported"
                if spectral["top_tenth_energy_fraction"] >= 0.9
                else "rejected"
            ),
            exact=False,
            facts={**base, **spectral},
        )
    )
    return claims, {"tensor": base["tensor"], "rank": 1, **spectral}


def _analyze_matrix(
    matrix: np.ndarray,
    *,
    base: dict,
    threshold: float,
    exhaustive: bool,
    low_rank_ratio: float,
    sparse_density: float,
) -> tuple[list[dict], dict]:
    rows, columns = matrix.shape
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank_threshold = max(
        threshold,
        (
            float(singular_values[0]) * max(rows, columns) * np.finfo(np.float64).eps
            if singular_values.size
            else 0.0
        ),
    )
    numerical_rank = int(np.count_nonzero(singular_values > rank_threshold))
    maximum_rank = min(rows, columns)
    rank_ratio = numerical_rank / max(1, maximum_rank)
    claims = [
        claim(
            kind="low_rank",
            status="supported" if rank_ratio <= low_rank_ratio else "rejected",
            exact=False,
            facts={
                **base,
                "matrix_shape": [rows, columns],
                "numerical_rank": numerical_rank,
                "maximum_rank": maximum_rank,
                "rank_ratio": rank_ratio,
                "rank_tolerance": rank_threshold,
                "maximum_rank_ratio": low_rank_ratio,
            },
        )
    ]

    row_unique = _unique_axis_count(matrix, axis=0, threshold=threshold)
    column_unique = _unique_axis_count(matrix, axis=1, threshold=threshold)
    exact_row_unique = _unique_axis_count(matrix, axis=0, threshold=0.0)
    exact_column_unique = _unique_axis_count(matrix, axis=1, threshold=0.0)
    claims.extend(
        [
            claim(
                kind="repeated_rows",
                status="supported" if row_unique < rows else "rejected",
                exact=exhaustive and (row_unique == rows or exact_row_unique < rows),
                facts={
                    **base,
                    "row_count": rows,
                    "unique_row_count": row_unique,
                    "tolerance": threshold,
                },
            ),
            claim(
                kind="repeated_columns",
                status="supported" if column_unique < columns else "rejected",
                exact=exhaustive
                and (column_unique == columns or exact_column_unique < columns),
                facts={
                    **base,
                    "column_count": columns,
                    "unique_column_count": column_unique,
                    "tolerance": threshold,
                },
            ),
        ]
    )

    block_facts = _repeated_block_facts(matrix, threshold)
    exact_block_facts = _repeated_block_facts(matrix, 0.0)
    claims.append(
        claim(
            kind="repeated_blocks",
            status=(
                "supported"
                if block_facts["best_repeated_block_ratio"] > 0
                else "rejected"
            ),
            exact=exhaustive
            and (
                block_facts["best_repeated_block_ratio"] == 0
                or exact_block_facts["best_repeated_block_ratio"] > 0
            ),
            facts={**base, **block_facts},
        )
    )

    nonzero = np.abs(matrix) > threshold
    lower_band, upper_band = _bandwidths(nonzero)
    exact_lower_band, exact_upper_band = _bandwidths(matrix != 0)
    maximum_bandwidth = max(1, max(rows, columns) // 4)
    banded = lower_band <= maximum_bandwidth and upper_band <= maximum_bandwidth
    claims.append(
        claim(
            kind="banded_structure",
            status="supported" if banded else "rejected",
            exact=exhaustive
            and (
                not banded
                or (exact_lower_band == lower_band and exact_upper_band == upper_band)
            ),
            facts={
                **base,
                "lower_bandwidth": lower_band,
                "upper_bandwidth": upper_band,
                "maximum_supported_bandwidth": maximum_bandwidth,
                "tolerance": threshold,
            },
        )
    )

    block_diagonal = _block_diagonal_facts(nonzero)
    exact_block_diagonal = _block_diagonal_facts(matrix != 0)
    claims.append(
        claim(
            kind="block_diagonal_structure",
            status=("supported" if block_diagonal["block_count"] >= 2 else "rejected"),
            exact=exhaustive
            and (
                block_diagonal["block_count"] < 2
                or exact_block_diagonal["block_count"] >= 2
            ),
            facts={**base, **block_diagonal, "tolerance": threshold},
        )
    )

    toeplitz_error = _toeplitz_error(matrix)
    circulant_error = _circulant_error(matrix)
    claims.extend(
        [
            claim(
                kind="toeplitz_structure",
                status="supported" if toeplitz_error <= threshold else "rejected",
                exact=exhaustive
                and (toeplitz_error > threshold or toeplitz_error == 0),
                facts={
                    **base,
                    "maximum_diagonal_error": toeplitz_error,
                    "tolerance": threshold,
                },
            ),
            claim(
                kind="circulant_structure",
                status=(
                    "supported"
                    if circulant_error is not None
                    and circulant_error <= threshold
                    else "rejected"
                ),
                exact=exhaustive
                and (
                    circulant_error is None
                    or circulant_error > threshold
                    or circulant_error == 0
                ),
                facts={
                    **base,
                    "maximum_cyclic_row_error": circulant_error,
                    "square_matrix": circulant_error is not None,
                    "tolerance": threshold,
                },
            ),
        ]
    )

    orthogonality = _orthogonality_facts(matrix)
    claims.append(
        claim(
            kind="orthogonality_invariant",
            status=(
                "supported"
                if orthogonality["maximum_normalized_off_diagonal"]
                <= max(
                    threshold,
                    1e-6,
                )
                else "rejected"
            ),
            exact=False,
            facts={**base, **orthogonality, "tolerance": threshold},
        )
    )

    kronecker = _kronecker_facts(matrix, threshold)
    claims.append(
        claim(
            kind="kronecker_tensor_product",
            status="supported" if kronecker is not None else "rejected",
            exact=False,
            facts={
                **base,
                "factorization": kronecker,
                "tolerance": threshold,
            },
        )
    )

    tensor_train = _tensor_train_facts(matrix, threshold)
    claims.append(
        claim(
            kind="tensor_train_structure",
            status=(
                "supported"
                if tensor_train["maximum_rank"] < min(matrix.shape)
                else "rejected"
            ),
            exact=False,
            facts={**base, **tensor_train, "tolerance": threshold},
        )
    )

    butterfly = _butterfly_facts(nonzero)
    exact_butterfly = _butterfly_facts(matrix != 0)
    claims.append(
        claim(
            kind="butterfly_structure",
            status="supported" if butterfly["compatible"] else "rejected",
            exact=exhaustive
            and (not butterfly["compatible"] or exact_butterfly["compatible"]),
            facts={**base, **butterfly},
        )
    )

    spectral = _spectral_facts(matrix)
    claims.append(
        claim(
            kind="spectral_structure",
            status=(
                "supported"
                if spectral["top_tenth_energy_fraction"] >= 0.9
                else "rejected"
            ),
            exact=False,
            facts={**base, **spectral},
        )
    )

    basis = _structured_basis_residual(matrix, singular_values, threshold)
    claims.append(
        claim(
            kind="structured_basis_sparse_exception",
            status=(
                "supported"
                if basis["residual_density"] <= sparse_density
                and basis["basis_rank"] < maximum_rank
                else "rejected"
            ),
            exact=False,
            facts={
                **base,
                **basis,
                "maximum_residual_density": sparse_density,
            },
        )
    )
    return claims, {
        "tensor": base["tensor"],
        "matrix_shape": [rows, columns],
        "numerical_rank": numerical_rank,
        "singular_values": singular_values[: min(32, singular_values.size)],
        **block_facts,
        **spectral,
    }


def _analyze_tensor_topology(
    context: ScopeAnalysisContext,
    values: np.ndarray,
    base: dict,
    threshold: float,
    exhaustive: bool,
) -> tuple[list[dict], dict]:
    convolution_nodes = [
        str(node["id"])
        for node in context.nodes
        if "conv" in str(node.get("op", "")).casefold()
        or "convolution" in str(node.get("attrs", {})).casefold()
    ]
    convolutional = bool(convolution_nodes) and values.ndim >= 2
    claims = [
        claim(
            kind="convolutional_structure",
            status="supported" if convolutional else "rejected",
            exact=convolutional,
            facts={
                **base,
                "source_node_ids": convolution_nodes,
                "kernel_shape": list(values.shape),
                "semantic_operator_evidence": convolutional,
            },
        )
    ]
    if values.ndim >= 3 and max(values.shape) <= 128:
        ranks = []
        for split in range(1, values.ndim):
            unfolding = values.reshape(
                math.prod(values.shape[:split]),
                math.prod(values.shape[split:]),
            )
            ranks.append(int(np.linalg.matrix_rank(unfolding, tol=threshold)))
        claims.append(
            claim(
                kind="tensor_train_structure",
                status=(
                    "supported"
                    if max(ranks, default=0) < max(values.shape)
                    else "rejected"
                ),
                exact=False,
                facts={**base, "unfolding_ranks": ranks, "tolerance": threshold},
            )
        )
    return claims, {
        "tensor": base["tensor"],
        "tensor_shape": list(values.shape),
        "convolution_nodes": convolution_nodes,
    }


def _inconclusive_decomposition_claims(
    base: dict,
    shape: tuple[int, int],
    limit: int,
) -> list[dict]:
    facts = {
        **base,
        "matrix_shape": list(shape),
        "reason": "observed matrix exceeds declared decomposition dimension budget",
        "decomposition_dimension_limit": limit,
    }
    return [
        claim(kind=kind, status="inconclusive", exact=False, facts=facts)
        for kind in (
            "low_rank",
            "shared_subspace",
            "kronecker_tensor_product",
            "tensor_train_structure",
            "structured_basis_sparse_exception",
        )
    ]


def _unique_axis_count(
    matrix: np.ndarray,
    *,
    axis: int,
    threshold: float,
) -> int:
    values = matrix if axis == 0 else matrix.T
    if threshold > 0:
        values = np.rint(values / threshold).astype(np.int64)
    contiguous = np.ascontiguousarray(values)
    return int(np.unique(contiguous, axis=0).shape[0])


def _repeated_block_facts(matrix: np.ndarray, threshold: float) -> dict:
    rows, columns = matrix.shape
    best = {"block_shape": None, "best_repeated_block_ratio": 0.0}
    for block_rows in (1, 2, 4, 8, 16):
        for block_columns in (1, 2, 4, 8, 16):
            if rows % block_rows or columns % block_columns:
                continue
            blocks = (
                matrix.reshape(
                    rows // block_rows,
                    block_rows,
                    columns // block_columns,
                    block_columns,
                )
                .transpose(0, 2, 1, 3)
                .reshape(-1, block_rows * block_columns)
            )
            unique = _unique_axis_count(blocks, axis=0, threshold=threshold)
            ratio = 1.0 - unique / max(1, blocks.shape[0])
            if ratio > best["best_repeated_block_ratio"]:
                best = {
                    "block_shape": [block_rows, block_columns],
                    "block_count": int(blocks.shape[0]),
                    "unique_block_count": unique,
                    "best_repeated_block_ratio": ratio,
                }
    return best


def _bandwidths(nonzero: np.ndarray) -> tuple[int, int]:
    locations = np.argwhere(nonzero)
    if not locations.size:
        return 0, 0
    offsets = locations[:, 0] - locations[:, 1]
    return (
        int(max(0, np.max(offsets))),
        int(max(0, np.max(-offsets))),
    )


def _block_diagonal_facts(nonzero: np.ndarray) -> dict:
    rows, columns = nonzero.shape
    maximum_blocks = min(rows, columns, 32)
    for block_count in range(maximum_blocks, 1, -1):
        if rows % block_count or columns % block_count:
            continue
        row_block = rows // block_count
        column_block = columns // block_count
        valid = True
        for row_group in range(block_count):
            for column_group in range(block_count):
                if row_group == column_group:
                    continue
                if np.any(
                    nonzero[
                        row_group * row_block : (row_group + 1) * row_block,
                        column_group * column_block : (column_group + 1) * column_block,
                    ]
                ):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return {
                "block_count": block_count,
                "block_shape": [row_block, column_block],
            }
    return {"block_count": 1, "block_shape": [rows, columns]}


def _toeplitz_error(matrix: np.ndarray) -> float:
    errors = []
    rows, columns = matrix.shape
    for offset in range(-(rows - 1), columns):
        diagonal = np.diagonal(matrix, offset=offset)
        if diagonal.size:
            errors.append(float(np.max(np.abs(diagonal - diagonal[0]))))
    return max(errors, default=0.0)


def _circulant_error(matrix: np.ndarray) -> float | None:
    if matrix.shape[0] != matrix.shape[1] or not matrix.size:
        return None
    return max(
        (
            float(np.max(np.abs(matrix[row] - np.roll(matrix[0], row))))
            for row in range(matrix.shape[0])
        ),
        default=0.0,
    )


def _orthogonality_facts(matrix: np.ndarray) -> dict:
    basis = matrix if matrix.shape[0] <= matrix.shape[1] else matrix.T
    gram = basis @ basis.T
    diagonal = np.diag(gram)
    normalizer = max(float(np.max(np.abs(diagonal))), np.finfo(float).eps)
    off_diagonal = gram - np.diag(diagonal)
    return {
        "tested_axis": "rows" if basis is matrix else "columns",
        "maximum_normalized_off_diagonal": (
            float(np.max(np.abs(off_diagonal))) / normalizer
        ),
        "minimum_squared_norm": float(np.min(diagonal)),
        "maximum_squared_norm": float(np.max(diagonal)),
    }


def _kronecker_facts(
    matrix: np.ndarray,
    threshold: float,
) -> dict | None:
    rows, columns = matrix.shape
    row_factors = [
        factor for factor in range(2, min(rows, 16) + 1) if rows % factor == 0
    ]
    column_factors = [
        factor for factor in range(2, min(columns, 16) + 1) if columns % factor == 0
    ]
    best = None
    for left_rows in row_factors:
        right_rows = rows // left_rows
        for left_columns in column_factors:
            right_columns = columns // left_columns
            rearranged = (
                matrix.reshape(
                    left_rows,
                    right_rows,
                    left_columns,
                    right_columns,
                )
                .transpose(0, 2, 1, 3)
                .reshape(left_rows * left_columns, right_rows * right_columns)
            )
            singular = np.linalg.svd(rearranged, compute_uv=False)
            residual = float(np.linalg.norm(singular[1:])) if singular.size > 1 else 0.0
            relative = residual / max(
                float(np.linalg.norm(singular)),
                np.finfo(float).eps,
            )
            candidate = {
                "left_shape": [left_rows, left_columns],
                "right_shape": [right_rows, right_columns],
                "relative_residual": relative,
            }
            if best is None or relative < best["relative_residual"]:
                best = candidate
    if best is None or best["relative_residual"] > max(threshold, 1e-7):
        return None
    return best


def _tensor_train_facts(matrix: np.ndarray, threshold: float) -> dict:
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.count_nonzero(singular > threshold))
    return {
        "unfolding_ranks": [rank],
        "maximum_rank": rank,
    }


def _butterfly_facts(nonzero: np.ndarray) -> dict:
    rows, columns = nonzero.shape
    square_power_two = rows == columns and rows > 0 and rows & (rows - 1) == 0
    row_degrees = np.count_nonzero(nonzero, axis=1)
    maximum_degree = int(np.max(row_degrees)) if row_degrees.size else 0
    allowed = int(math.log2(rows)) + 1 if square_power_two else 0
    return {
        "compatible": square_power_two and maximum_degree <= allowed,
        "square_power_of_two": square_power_two,
        "maximum_row_degree": maximum_degree,
        "maximum_butterfly_row_degree": allowed,
    }


def _spectral_facts(values: np.ndarray) -> dict:
    spectrum = np.fft.rfft(values.reshape(-1))
    energy = np.abs(spectrum) ** 2
    total = float(np.sum(energy))
    keep = max(1, math.ceil(energy.size * 0.1))
    top = float(np.sum(np.partition(energy, -keep)[-keep:]))
    return {
        "coefficient_count": int(energy.size),
        "top_tenth_coefficient_count": keep,
        "top_tenth_energy_fraction": top / total if total else 1.0,
    }


def _structured_basis_residual(
    matrix: np.ndarray,
    singular_values: np.ndarray,
    threshold: float,
) -> dict:
    maximum_rank = min(matrix.shape)
    basis_rank = max(1, min(maximum_rank - 1, math.ceil(maximum_rank * 0.25)))
    if maximum_rank <= 1:
        return {"basis_rank": maximum_rank, "residual_density": 0.0}
    u, _singular, vh = np.linalg.svd(matrix, full_matrices=False)
    approximation = (u[:, :basis_rank] * singular_values[:basis_rank][None, :]) @ vh[
        :basis_rank
    ]
    residual = matrix - approximation
    return {
        "basis_rank": basis_rank,
        "residual_density": float(np.count_nonzero(np.abs(residual) > threshold))
        / residual.size,
        "relative_residual_norm": float(np.linalg.norm(residual))
        / max(float(np.linalg.norm(matrix)), np.finfo(float).eps),
    }
