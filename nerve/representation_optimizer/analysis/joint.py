from __future__ import annotations

from collections import defaultdict
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
from nerve.representation_optimizer.analysis.decomposition import exact_matrix_svd


class JointParameterAnalyzer:
    """Find structure that is visible only across related parameters."""

    analyzer_id = "joint_parameter_structure"
    version = "2"

    def analyze(self, context: ScopeAnalysisContext) -> AnalyzerResult:
        records = [
            _parameter_record(context, parameter) for parameter in context.parameters
        ]
        claims, pair_details, search_details = _search_relationships(context, records)
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
                "search": search_details,
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
    return {
        "binding": binding,
        "observation": observation,
        "parameter_nodes": _parameter_nodes(context, binding),
        **context.computations.get_or_compute(
            "joint_parameter_facts",
            (
                context.computations.observation_identity(observation),
                threshold,
            ),
            lambda: _parameter_facts(values, threshold),
        ),
    }


def _parameter_facts(
    values: np.ndarray,
    threshold: float,
) -> dict:
    matrix = values.reshape(-1, values.shape[-1]) if values.ndim >= 2 else None
    return {
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


def _search_relationships(
    context: ScopeAnalysisContext,
    records: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    pair_count = len(records) * (len(records) - 1) // 2
    discoveries: dict[tuple[int, int], dict] = defaultdict(dict)
    search_counts = {
        "duplicate_pairs": pair_count,
        "coordinate_compatible_pairs": 0,
        "affine_compatible_pairs": 0,
        "proper_subspace_pairs": 0,
        "coupled_shape_compatible_pairs": 0,
    }

    for group in _groups(records, lambda record: record["raw_digest"]):
        for left, right in combinations(group, 2):
            discoveries[(left, right)]["exact_duplicate"] = True

    coordinate_pairs: set[tuple[int, int]] = set()
    for axis, digest_field in (
        ("independent_row_permutation", "row_canonical_digest"),
        ("independent_column_permutation", "column_canonical_digest"),
    ):
        for group in _groups(
            records,
            lambda record, field=digest_field: (
                tuple(record["values"].shape),
                record[field],
            )
            if record["values"].ndim >= 2
            else None,
        ):
            for left, right in combinations(group, 2):
                pair = (left, right)
                coordinate_pairs.add(pair)
                discoveries[pair].setdefault(
                    "coordinate",
                    {
                        "kind": axis,
                        "canonical_digest": records[left][digest_field],
                    },
                )
    search_counts["coordinate_compatible_pairs"] = sum(
        len(group) * (len(group) - 1) // 2
        for group in _groups(
            records,
            lambda record: tuple(record["values"].shape)
            if record["values"].ndim >= 2
            else None,
        )
    )
    for left, right in coordinate_pairs:
        records_for_pair = (records[left], records[right])
        exact = any(
            first[field] == second[field]
            for field in (
                "exact_row_canonical_digest",
                "exact_column_canonical_digest",
            )
            for first, second in (records_for_pair,)
        )
        discoveries[(left, right)]["exact_coordinate"] = exact

    affine, affine_pair_count = _affine_generators(context, records)
    search_counts["affine_compatible_pairs"] = affine_pair_count
    for pair, generator in affine.items():
        discoveries[pair]["generator"] = generator

    subspaces, subspace_pair_count = _proper_shared_subspaces(context, records)
    search_counts["proper_subspace_pairs"] = subspace_pair_count
    for pair, subspace in subspaces.items():
        discoveries[pair]["subspace"] = subspace

    for left_index, right_index in combinations(range(len(records)), 2):
        left = records[left_index]
        right = records[right_index]
        left_values = left["values"]
        right_values = right["values"]
        if (
            left_values.ndim != 2
            or right_values.ndim != 2
            or left_values.shape[0] != right_values.shape[1]
        ):
            continue
        search_counts["coupled_shape_compatible_pairs"] += 1
        symmetry = _coupled_coordinate_symmetry(left, right)
        if symmetry is not None:
            discoveries[(left_index, right_index)]["coupled_symmetry"] = symmetry

    claims = []
    details = []
    discovery_counts: defaultdict[str, int] = defaultdict(int)
    for (left_index, right_index), found in sorted(discoveries.items()):
        left = records[left_index]
        right = records[right_index]
        base = _pair_base(context, left, right)
        both_exhaustive = (
            left["observation"].exhaustive and right["observation"].exhaustive
        )
        exact_duplicate = found.get("exact_duplicate", False)
        coordinate = found.get("coordinate")
        exact_coordinate = found.get("exact_coordinate", False)
        generator = found.get("generator")
        subspace = found.get("subspace")
        coupled_symmetry = found.get("coupled_symmetry")
        if exact_duplicate:
            discovery_counts["common_parameter_subexpression"] += 1
            claims.append(
                claim(
                    kind="common_parameter_subexpression",
                    status="supported",
                    exact=both_exhaustive,
                    facts={
                        **base,
                        "same_observed_shape": True,
                        "left_digest": left["raw_digest"],
                        "right_digest": right["raw_digest"],
                    },
                )
            )
        if coordinate is not None:
            discovery_counts["coordinate_equivalence"] += 1
            discovery_counts["permutation_symmetry"] += 1
            claims.extend(
                (
                    claim(
                        kind="coordinate_equivalence",
                        status="supported",
                        exact=both_exhaustive and exact_coordinate,
                        facts={**base, "equivalence": coordinate},
                    ),
                    claim(
                        kind="permutation_symmetry",
                        status="supported",
                        exact=both_exhaustive and exact_coordinate,
                        facts={
                            **base,
                            "canonicalization": coordinate,
                            "raw_values_equal": exact_duplicate,
                        },
                    ),
                )
            )
        if generator is not None:
            discovery_counts["shared_parameter_generator"] += 1
            claims.append(
                claim(
                    kind="shared_parameter_generator",
                    status="supported",
                    exact=both_exhaustive and generator["maximum_error"] == 0,
                    facts={**base, "affine_generator": generator},
                )
            )
        if subspace is not None:
            discovery_counts["shared_subspace"] += 1
            claims.append(
                claim(
                    kind="shared_subspace",
                    status="supported",
                    exact=False,
                    facts={**base, "subspace": subspace},
                )
            )
        if coupled_symmetry is not None:
            discovery_counts["coupled_coordinate_permutation_symmetry"] += 1
            claims.append(
                claim(
                    kind="coupled_coordinate_permutation_symmetry",
                    status="supported",
                    exact=True,
                    facts={**base, "symmetry": coupled_symmetry},
                )
            )
        same_role = left["binding"].semantic_role == right["binding"].semantic_role
        motif = same_role and (exact_duplicate or coordinate is not None)
        if motif:
            discovery_counts["cross_component_motif"] += 1
            claims.append(
                claim(
                    kind="cross_component_motif",
                    status="supported",
                    exact=both_exhaustive
                    and (exact_duplicate or exact_coordinate),
                    facts={
                        **base,
                        "same_semantic_role": True,
                        "same_parameter_values": exact_duplicate,
                        "coordinate_equivalent": coordinate is not None,
                    },
                )
            )
        details.append(
            {
                **base,
                "exact_duplicate": exact_duplicate,
                "coordinate_equivalence": coordinate,
                "affine_generator": generator,
                "shared_subspace": subspace,
                "coupled_coordinate_symmetry": coupled_symmetry,
            }
        )

    search_details = {
        "parameter_count": len(records),
        "searched_pair_count": pair_count,
        "relationship_pair_count": len(discoveries),
        "pair_without_relationship_count": pair_count - len(discoveries),
        "compatible_pair_counts": search_counts,
        "discovery_counts": dict(sorted(discovery_counts.items())),
    }
    claims.insert(
        0,
        claim(
            kind="joint_parameter_search_coverage",
            status="supported" if pair_count else "rejected",
            exact=True,
            facts=search_details,
        ),
    )
    return claims, details, search_details


def _pair_base(
    context: ScopeAnalysisContext,
    left: dict,
    right: dict,
) -> dict:
    return {
        "left_tensor": left["binding"].tensor_name,
        "right_tensor": right["binding"].tensor_name,
        "left_semantic_role": left["binding"].semantic_role,
        "right_semantic_role": right["binding"].semantic_role,
        "left_observation": context.observation_domain(left["binding"].tensor_name),
        "right_observation": context.observation_domain(right["binding"].tensor_name),
    }


def _groups(
    records: list[dict],
    key,
) -> list[list[int]]:
    grouped: defaultdict[object, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        value = key(record)
        if value is not None:
            grouped[value].append(index)
    return [indices for indices in grouped.values() if len(indices) >= 2]


def _shape_groups(
    records: list[dict],
    *,
    minimum_size: int,
) -> list[list[int]]:
    return _groups(
        records,
        lambda record: tuple(record["values"].shape)
        if record["values"].size >= minimum_size
        else None,
    )


def _affine_generators(
    context: ScopeAnalysisContext,
    records: list[dict],
) -> tuple[dict[tuple[int, int], dict], int]:
    key = tuple(
        (
            context.computations.observation_identity(record["observation"]),
            record["threshold"],
        )
        for record in records
    )
    return context.computations.get_or_compute(
        "joint_affine_search",
        key,
        lambda: _compute_affine_generators(records),
    )


def _compute_affine_generators(
    records: list[dict],
) -> tuple[dict[tuple[int, int], dict], int]:
    generators = {}
    searched_pair_count = 0
    for group in _shape_groups(records, minimum_size=2):
        searched_pair_count += len(group) * (len(group) - 1) // 2
        values = np.stack(
            [records[index]["values"].reshape(-1) for index in group],
        )
        element_count = values.shape[1]
        means = np.mean(values, axis=1)
        gram = values @ values.T
        centered_gram = gram - element_count * np.outer(means, means)
        for left_position, right_position in combinations(range(len(group)), 2):
            left_index = group[left_position]
            right_index = group[right_position]
            left_record = records[left_index]
            right_record = records[right_index]
            threshold = max(
                left_record["threshold"],
                right_record["threshold"],
            )
            computational_tolerance = _affine_computational_tolerance(
                left_record["values"],
                right_record["values"],
                threshold,
            )
            left_variance = max(
                0.0,
                float(centered_gram[left_position, left_position]),
            )
            right_variance = max(
                0.0,
                float(centered_gram[right_position, right_position]),
            )
            covariance = float(centered_gram[left_position, right_position])
            residual_squared = (
                right_variance
                if left_variance == 0
                else max(
                    0.0,
                    right_variance - covariance * covariance / left_variance,
                )
            )
            numerical_slack = (
                np.finfo(np.float64).eps
                * max(1.0, right_variance, abs(covariance))
                * element_count
                * 64
            )
            maximum_possible_residual = (
                element_count * computational_tolerance**2 + numerical_slack
            )
            if residual_squared > maximum_possible_residual:
                continue
            generator = _affine_generator(
                left_record["values"],
                right_record["values"],
                threshold,
            )
            if generator is not None:
                generators[(left_index, right_index)] = generator
    return generators, searched_pair_count


def _affine_computational_tolerance(
    left: np.ndarray,
    right: np.ndarray,
    threshold: float,
) -> float:
    return max(
        threshold,
        np.finfo(np.float64).eps
        * max(
            1.0,
            float(np.max(np.abs(left))),
            float(np.max(np.abs(right))),
        )
        * max(1, left.size)
        * 16,
    )


def _proper_shared_subspaces(
    context: ScopeAnalysisContext,
    records: list[dict],
) -> tuple[dict[tuple[int, int], dict], int]:
    bases: dict[int, np.ndarray] = {}
    ambient_groups: defaultdict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        values = record["values"]
        if values.ndim < 2:
            continue
        matrix = values.reshape(-1, values.shape[-1])
        if max(matrix.shape) > context.budget.decomposition_dimension_limit:
            continue
        basis = _dominant_row_basis(
            context,
            record["binding"].tensor_name,
            matrix,
            record["threshold"],
        )
        ambient_dimension = matrix.shape[1]
        if not basis.size or basis.shape[0] >= ambient_dimension:
            continue
        bases[index] = basis
        ambient_groups[ambient_dimension].append(index)

    found = {}
    searched_pair_count = 0
    for ambient_dimension, group in ambient_groups.items():
        searched_pair_count += len(group) * (len(group) - 1) // 2
        for left_index, right_index in combinations(group, 2):
            left_basis = bases[left_index]
            right_basis = bases[right_index]
            singular = np.linalg.svd(
                left_basis @ right_basis.T,
                compute_uv=False,
            )
            overlap = float(np.mean(singular**2))
            if overlap < 0.9:
                continue
            found[(left_index, right_index)] = {
                "ambient_dimension": ambient_dimension,
                "left_rank": int(left_basis.shape[0]),
                "right_rank": int(right_basis.shape[0]),
                "overlap": overlap,
                "minimum_principal_cosine": float(np.min(singular)),
            }
    return found, searched_pair_count


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
    computational_tolerance = _affine_computational_tolerance(
        left,
        right,
        threshold,
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


def _coupled_coordinate_symmetry(
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
    left_nodes = left["parameter_nodes"]
    right_nodes = right["parameter_nodes"]
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


def _dominant_row_basis(
    context: ScopeAnalysisContext,
    tensor_name: str,
    matrix: np.ndarray,
    threshold: float,
) -> np.ndarray:
    decomposition = exact_matrix_svd(context, tensor_name, matrix)
    numerical_threshold = max(
        threshold,
        (
            float(decomposition.singular_values[0])
            * max(matrix.shape)
            * np.finfo(np.float64).eps
            if decomposition.singular_values.size
            else 0.0
        ),
    )
    rank = int(
        np.count_nonzero(decomposition.singular_values > numerical_threshold)
    )
    return decomposition.right_vectors[:rank]


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
