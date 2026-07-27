from __future__ import annotations

import numpy as np

from nerve.representation_optimizer.analysis.claims import (
    AnalyzerResult,
    claim,
    observation_facts,
    tolerance_threshold,
)
from nerve.representation_optimizer.analysis.context import ScopeAnalysisContext


class ElementwiseStructureAnalyzer:
    analyzer_id = "elementwise_structure"
    version = "1"

    def analyze(self, context: ScopeAnalysisContext) -> AnalyzerResult:
        claims = []
        tensor_details = []
        for parameter in context.parameters:
            observation = context.observation(parameter.tensor_name)
            values = observation.values
            threshold = tolerance_threshold(
                values,
                absolute_tolerance=context.budget.absolute_tolerance,
                relative_tolerance=context.budget.relative_tolerance,
            )
            intrinsic_claims, intrinsic_details = (
                context.computations.get_or_compute(
                    "elementwise_parameter_facts",
                    (
                        context.computations.observation_identity(observation),
                        threshold,
                        context.budget.sparse_density_threshold,
                        context.budget.codebook_ratio_threshold,
                    ),
                    lambda: _analyze_parameter(
                        context,
                        parameter.tensor_name,
                        values,
                        threshold,
                        observation.exhaustive,
                    ),
                )
            )
            claims.extend(
                {
                    **item,
                    "facts": {
                        **item["facts"],
                        "semantic_role": parameter.semantic_role,
                    },
                }
                for item in intrinsic_claims
            )
            tensor_details.append(
                {
                    **intrinsic_details,
                    "semantic_role": parameter.semantic_role,
                }
            )
        return AnalyzerResult(
            claims=tuple(claims),
            details={"tensors": tensor_details},
        )


def _analyze_parameter(
    context: ScopeAnalysisContext,
    tensor_name: str,
    values: np.ndarray,
    threshold: float,
    exhaustive: bool,
) -> tuple[list[dict], dict]:
    flat = values.reshape(-1)
    zero_mask = np.abs(flat) <= threshold
    zero_count = int(np.count_nonzero(zero_mask))
    density = float((flat.size - zero_count) / flat.size) if flat.size else 0.0
    all_zero = zero_count == flat.size
    exactly_zero = bool(np.all(flat == 0))
    zero_status = (
        "supported"
        if all_zero and exhaustive
        else "inconclusive"
        if all_zero
        else "rejected"
    )
    base = observation_facts(context, tensor_name)
    claims = [
        claim(
            kind="zero_parameter",
            status=zero_status,
            exact=exhaustive and (zero_status == "rejected" or exactly_zero),
            facts={
                **base,
                "zero_threshold": threshold,
                "observed_zero_count": zero_count,
                "observed_density": density,
            },
        )
    ]

    first = flat[0] if flat.size else np.float32(0)
    deviations = np.abs(flat - first)
    constant = bool(np.all(deviations <= threshold))
    exactly_constant = bool(np.all(flat == first))
    constant_status = (
        "supported"
        if constant and exhaustive
        else "inconclusive"
        if constant
        else "rejected"
    )
    claims.append(
        claim(
            kind="constant_parameter",
            status=constant_status,
            exact=exhaustive
            and (constant_status == "rejected" or exactly_constant),
            facts={
                **base,
                "candidate_value": float(first),
                "maximum_observed_deviation": (
                    float(np.max(deviations)) if deviations.size else 0.0
                ),
                "tolerance": threshold,
            },
        )
    )

    claims.append(
        claim(
            kind="sparse_parameter",
            status=(
                "supported"
                if density <= context.budget.sparse_density_threshold
                else "rejected"
            ),
            exact=exhaustive
            and (
                density > context.budget.sparse_density_threshold
                or (
                    float(np.count_nonzero(flat) / max(1, flat.size))
                    <= context.budget.sparse_density_threshold
                )
            ),
            facts={
                **base,
                "observed_density": density,
                "maximum_density": context.budget.sparse_density_threshold,
                "zero_threshold": threshold,
            },
        )
    )

    unique, counts = np.unique(flat, return_counts=True)
    probabilities = counts.astype(np.float64) / max(1, flat.size)
    entropy_bits = float(
        -np.sum(
            probabilities
            * np.log2(
                probabilities,
                out=np.zeros_like(probabilities),
                where=probabilities > 0,
            )
        )
    )
    raw_bits = max(1, values.dtype.itemsize * 8)
    normalized_entropy = entropy_bits / raw_bits
    codebook_ratio = unique.size / max(1, flat.size)
    codebook_supported = codebook_ratio <= context.budget.codebook_ratio_threshold
    common_count = int(np.max(counts)) if counts.size else 0
    repeated_ratio = common_count / max(1, flat.size)
    common_value = float(unique[int(np.argmax(counts))]) if unique.size else 0.0
    shared_facts = {
        **base,
        "observed_unique_value_count": int(unique.size),
        "observed_value_count": int(flat.size),
        "codebook_ratio": codebook_ratio,
        "shannon_entropy_bits_per_value": entropy_bits,
        "normalized_storage_entropy": normalized_entropy,
    }
    claims.append(
        claim(
            kind="low_entropy_codebook",
            status="supported" if codebook_supported else "rejected",
            exact=exhaustive,
            facts={
                **shared_facts,
                "maximum_codebook_ratio": context.budget.codebook_ratio_threshold,
            },
        )
    )
    claims.append(
        claim(
            kind="repeated_quantized_values",
            status=(
                "supported"
                if repeated_ratio >= 1 - context.budget.codebook_ratio_threshold
                else "rejected"
            ),
            exact=exhaustive,
            facts={
                **shared_facts,
                "most_common_value": common_value,
                "most_common_value_ratio": repeated_ratio,
            },
        )
    )
    return claims, {
        "tensor": tensor_name,
        "minimum": float(np.min(flat)) if flat.size else 0.0,
        "maximum": float(np.max(flat)) if flat.size else 0.0,
        "mean": float(np.mean(flat)) if flat.size else 0.0,
        "standard_deviation": float(np.std(flat)) if flat.size else 0.0,
        "finite": bool(np.all(np.isfinite(flat))),
        "observed_unique_values": int(unique.size),
        "entropy_bits": entropy_bits,
    }
