from __future__ import annotations

from itertools import combinations

import numpy as np

from nerve.representation_optimizer.analysis.claims import (
    AnalyzerResult,
    array_digest,
    claim,
    tolerance_threshold,
)
from nerve.representation_optimizer.analysis.context import (
    ParameterBinding,
    ScopeAnalysisContext,
)


class JointParameterAnalyzer:
    """Find structure that is visible only across related parameters."""

    analyzer_id = "joint_parameter_structure"
    version = "1"

    def analyze(self, context: ScopeAnalysisContext) -> AnalyzerResult:
        records = [
            _parameter_record(context, parameter) for parameter in context.parameters
        ]
        claims = []
        pair_details = []
        for left, right in combinations(records, 2):
            pair_claims, details = _analyze_pair(context, left, right)
            claims.extend(pair_claims)
            pair_details.append(details)
        expert_claims = []
        for record in records:
            expert_claims.extend(_repeated_expert_claims(context, record))
        claims.extend(expert_claims)
        return AnalyzerResult(
            claims=tuple(claims),
            details={
                "parameters": [
                    {
                        "tensor": record["binding"].tensor_name,
                        "semantic_role": record["binding"].semantic_role,
                        "raw_digest": record["raw_digest"],
                        "row_canonical_digest": record["row_canonical_digest"],
                        "column_canonical_digest": record["column_canonical_digest"],
                    }
                    for record in records
                ],
                "pairs": pair_details,
            },
        )


def _parameter_record(
    context: ScopeAnalysisContext,
    binding: ParameterBinding,
) -> dict:
    observation = context.observation(binding.tensor_name)
    values = observation.values.astype(np.float64, copy=False)
    threshold = tolerance_threshold(
        values,
        absolute_tolerance=context.budget.absolute_tolerance,
        relative_tolerance=context.budget.relative_tolerance,
    )
    matrix = values.reshape(-1, values.shape[-1]) if values.ndim >= 2 else None
    return {
        "binding": binding,
        "observation": observation,
        "values": values,
        "threshold": threshold,
        "raw_digest": array_digest(values),
        "row_canonical_digest": (
            array_digest(_canonicalize_axis(matrix, axis=0, threshold=threshold))
            if matrix is not None
            else None
        ),
        "column_canonical_digest": (
            array_digest(_canonicalize_axis(matrix, axis=1, threshold=threshold))
            if matrix is not None
            else None
        ),
        "exact_row_canonical_digest": (
            array_digest(_canonicalize_axis(matrix, axis=0, threshold=0.0))
            if matrix is not None
            else None
        ),
        "exact_column_canonical_digest": (
            array_digest(_canonicalize_axis(matrix, axis=1, threshold=0.0))
            if matrix is not None
            else None
        ),
    }


def _analyze_pair(
    context: ScopeAnalysisContext,
    left: dict,
    right: dict,
) -> tuple[list[dict], dict]:
    left_values = left["values"]
    right_values = right["values"]
    base = {
        "left_tensor": left["binding"].tensor_name,
        "right_tensor": right["binding"].tensor_name,
        "left_semantic_role": left["binding"].semantic_role,
        "right_semantic_role": right["binding"].semantic_role,
        "left_observation": context.observation_domain(left["binding"].tensor_name),
        "right_observation": context.observation_domain(right["binding"].tensor_name),
    }
    both_exhaustive = left["observation"].exhaustive and right["observation"].exhaustive
    same_shape = left_values.shape == right_values.shape
    exact_duplicate = same_shape and left["raw_digest"] == right["raw_digest"]
    claims = [
        claim(
            kind="common_parameter_subexpression",
            status="supported" if exact_duplicate else "rejected",
            exact=both_exhaustive,
            facts={
                **base,
                "same_observed_shape": same_shape,
                "left_digest": left["raw_digest"],
                "right_digest": right["raw_digest"],
            },
        )
    ]

    coordinate = _coordinate_equivalence(left, right)
    exact_coordinate = _coordinate_equivalence(
        {
            **left,
            "row_canonical_digest": left["exact_row_canonical_digest"],
            "column_canonical_digest": left["exact_column_canonical_digest"],
        },
        {
            **right,
            "row_canonical_digest": right["exact_row_canonical_digest"],
            "column_canonical_digest": right["exact_column_canonical_digest"],
        },
    )
    claims.append(
        claim(
            kind="coordinate_equivalence",
            status="supported" if coordinate is not None else "rejected",
            exact=both_exhaustive
            and (coordinate is None or exact_coordinate is not None),
            facts={**base, "equivalence": coordinate},
        )
    )
    claims.append(
        claim(
            kind="permutation_symmetry",
            status="supported" if coordinate is not None else "rejected",
            exact=both_exhaustive
            and (coordinate is None or exact_coordinate is not None),
            facts={
                **base,
                "canonicalization": coordinate,
                "raw_values_equal": exact_duplicate,
            },
        )
    )

    generator = _affine_generator(
        left_values,
        right_values,
        max(
            left["threshold"],
            right["threshold"],
        ),
    )
    claims.append(
        claim(
            kind="shared_parameter_generator",
            status="supported" if generator is not None else "rejected",
            exact=both_exhaustive
            and generator is not None
            and generator["maximum_error"] == 0,
            facts={**base, "affine_generator": generator},
        )
    )

    subspace = _shared_subspace(context, left, right)
    claims.append(
        claim(
            kind="shared_subspace",
            status=(
                "supported"
                if subspace is not None and subspace["overlap"] >= 0.9
                else "rejected"
            ),
            exact=False,
            facts={**base, "subspace": subspace},
        )
    )

    coupled_symmetry = _coupled_coordinate_symmetry(context, left, right)
    claims.append(
        claim(
            kind="coupled_coordinate_permutation_symmetry",
            status=("supported" if coupled_symmetry is not None else "rejected"),
            exact=True,
            facts={**base, "symmetry": coupled_symmetry},
        )
    )

    same_role = left["binding"].semantic_role == right["binding"].semantic_role
    motif = same_role and (exact_duplicate or coordinate is not None)
    claims.append(
        claim(
            kind="cross_component_motif",
            status="supported" if motif else "rejected",
            exact=both_exhaustive and motif,
            facts={
                **base,
                "same_semantic_role": same_role,
                "same_parameter_values": exact_duplicate,
                "coordinate_equivalent": coordinate is not None,
            },
        )
    )
    return claims, {
        **base,
        "same_shape": same_shape,
        "exact_duplicate": exact_duplicate,
        "coordinate_equivalence": coordinate,
        "affine_generator": generator,
        "shared_subspace": subspace,
        "coupled_coordinate_symmetry": coupled_symmetry,
    }


def _coordinate_equivalence(left: dict, right: dict) -> dict | None:
    if left["values"].ndim < 2 or left["values"].shape != right["values"].shape:
        return None
    if left["row_canonical_digest"] == right["row_canonical_digest"]:
        return {
            "kind": "independent_row_permutation",
            "canonical_digest": left["row_canonical_digest"],
        }
    if left["column_canonical_digest"] == right["column_canonical_digest"]:
        return {
            "kind": "independent_column_permutation",
            "canonical_digest": left["column_canonical_digest"],
        }
    return None


def _canonicalize_axis(
    matrix: np.ndarray,
    *,
    axis: int,
    threshold: float,
) -> np.ndarray:
    records = matrix if axis == 0 else matrix.T
    comparison = (
        np.rint(records / threshold).astype(np.int64) if threshold > 0 else records
    )
    keys = tuple(
        comparison[:, column] for column in range(comparison.shape[1] - 1, -1, -1)
    )
    order = np.lexsort(keys) if keys else np.arange(records.shape[0])
    canonical = records[order]
    return np.ascontiguousarray(canonical if axis == 0 else canonical.T)


def _affine_generator(
    left: np.ndarray,
    right: np.ndarray,
    threshold: float,
) -> dict | None:
    if left.shape != right.shape or left.size < 2:
        return None
    x = left.reshape(-1)
    y = right.reshape(-1)
    design = np.column_stack((x, np.ones_like(x)))
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ coefficients
    maximum_error = float(np.max(np.abs(predicted - y)))
    computational_tolerance = max(
        threshold,
        np.finfo(np.float64).eps
        * max(
            1.0,
            float(np.max(np.abs(x))),
            float(np.max(np.abs(y))),
        )
        * max(1, x.size)
        * 16,
    )
    if maximum_error > computational_tolerance:
        return None
    return {
        "scale": float(coefficients[0]),
        "offset": float(coefficients[1]),
        "maximum_error": maximum_error,
        "declared_tolerance": threshold,
        "computational_tolerance": computational_tolerance,
    }


def _shared_subspace(
    context: ScopeAnalysisContext,
    left: dict,
    right: dict,
) -> dict | None:
    if left["values"].ndim < 2 or right["values"].ndim < 2:
        return None
    left_matrix = left["values"].reshape(-1, left["values"].shape[-1])
    right_matrix = right["values"].reshape(-1, right["values"].shape[-1])
    if (
        left_matrix.shape[1] != right_matrix.shape[1]
        or max(left_matrix.shape + right_matrix.shape)
        > context.budget.decomposition_dimension_limit
    ):
        return None
    left_basis = _dominant_row_basis(left_matrix, left["threshold"])
    right_basis = _dominant_row_basis(right_matrix, right["threshold"])
    if not left_basis.size or not right_basis.size:
        return None
    singular = np.linalg.svd(left_basis @ right_basis.T, compute_uv=False)
    return {
        "left_rank": int(left_basis.shape[0]),
        "right_rank": int(right_basis.shape[0]),
        "overlap": float(np.mean(singular**2)),
        "minimum_principal_cosine": float(np.min(singular)),
    }


def _coupled_coordinate_symmetry(
    context: ScopeAnalysisContext,
    left: dict,
    right: dict,
) -> dict | None:
    left_values = left["values"]
    right_values = right["values"]
    if (
        left_values.ndim != 2
        or right_values.ndim != 2
        or left_values.shape[0] != right_values.shape[1]
    ):
        return None
    left_binding = left["binding"]
    right_binding = right["binding"]
    left_nodes = _parameter_nodes(context, left_binding)
    right_nodes = _parameter_nodes(context, right_binding)
    connected = []
    for source in left_nodes:
        source_outputs = set(source.get("outputs", []))
        for destination in right_nodes:
            shared = sorted(source_outputs.intersection(destination.get("inputs", [])))
            if shared:
                connected.append(
                    {
                        "source_node_id": source["id"],
                        "destination_node_id": destination["id"],
                        "shared_signals": shared,
                    }
                )
    if not connected:
        return None
    return {
        "coordinate_width": int(left_values.shape[0]),
        "source_axis": 0,
        "destination_axis": 1,
        "action": "permute source rows and destination columns by inverse maps",
        "connected_parameter_uses": connected,
    }


def _parameter_nodes(
    context: ScopeAnalysisContext,
    binding: ParameterBinding,
) -> list[dict]:
    result = []
    for node in context.nodes:
        if node.get("component_id") != binding.component_id:
            continue
        if node.get("op") != "linear":
            continue
        if binding.parameter_ref_id in node.get("params", []):
            result.append(node)
    return result


def _dominant_row_basis(matrix: np.ndarray, threshold: float) -> np.ndarray:
    _u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.count_nonzero(singular > threshold))
    return vh[:rank]


def _repeated_expert_claims(
    context: ScopeAnalysisContext,
    record: dict,
) -> list[dict]:
    values = record["values"]
    role = record["binding"].semantic_role.casefold()
    if values.ndim < 3 and "expert" not in role:
        return []
    experts = values if values.ndim >= 3 else values.reshape(values.shape[0], -1)
    flattened = experts.reshape(experts.shape[0], -1)
    canonical = _canonicalize_axis(
        flattened,
        axis=0,
        threshold=record["threshold"],
    )
    unique_count = int(np.unique(canonical, axis=0).shape[0])
    exact_unique_count = int(np.unique(flattened, axis=0).shape[0])
    return [
        claim(
            kind="repeated_experts",
            status=("supported" if unique_count < flattened.shape[0] else "rejected"),
            exact=record["observation"].exhaustive
            and (
                unique_count == flattened.shape[0]
                or exact_unique_count < flattened.shape[0]
            ),
            facts={
                "tensor": record["binding"].tensor_name,
                "semantic_role": record["binding"].semantic_role,
                "expert_axis": 0,
                "expert_count": int(flattened.shape[0]),
                "unique_expert_count": unique_count,
                "observation": context.observation_domain(
                    record["binding"].tensor_name
                ),
            },
        )
    ]
