from __future__ import annotations

import numpy as np

from nerve.representation_optimizer.analysis.claims import (
    AnalyzerResult,
    claim,
    observation_facts,
    tolerance_threshold,
)
from nerve.representation_optimizer.analysis.context import ScopeAnalysisContext


class ProceduralStructureAnalyzer:
    analyzer_id = "procedural_structure"
    version = "2"

    def analyze(self, context: ScopeAnalysisContext) -> AnalyzerResult:
        claims = []
        details = []
        for parameter in context.parameters:
            observation = context.observation(parameter.tensor_name)
            values = observation.values.reshape(-1).astype(np.float64)
            threshold = tolerance_threshold(
                values,
                absolute_tolerance=context.budget.absolute_tolerance,
                relative_tolerance=context.budget.relative_tolerance,
            )
            base = observation_facts(context, parameter.tensor_name)
            structure = context.computations.get_or_compute(
                "procedural_parameter_facts",
                (
                    context.computations.observation_identity(observation),
                    threshold,
                ),
                lambda: _procedural_facts(values, threshold),
            )
            period = structure["period"]
            exact_period = structure["exact_period"]
            claims.append(
                claim(
                    kind="periodic_parameter_generator",
                    status="supported" if period is not None else "rejected",
                    exact=observation.exhaustive
                    and (period is None or exact_period is not None),
                    facts={
                        **base,
                        "period": period,
                        "tolerance": threshold,
                    },
                )
            )

            affine = structure["affine_recurrence"]
            exact_affine = structure["exact_affine_recurrence"]
            claims.append(
                claim(
                    kind="affine_recurrence_generator",
                    status="supported" if affine is not None else "rejected",
                    exact=observation.exhaustive
                    and (affine is None or exact_affine is not None),
                    facts={
                        **base,
                        "recurrence": affine,
                        "tolerance": threshold,
                    },
                )
            )

            polynomial_degree = structure["polynomial_degree"]
            exact_polynomial_degree = structure["exact_polynomial_degree"]
            claims.append(
                claim(
                    kind="polynomial_parameter_generator",
                    status=(
                        "supported" if polynomial_degree is not None else "rejected"
                    ),
                    exact=observation.exhaustive
                    and (
                        polynomial_degree is None or exact_polynomial_degree is not None
                    ),
                    facts={
                        **base,
                        "degree": polynomial_degree,
                        "tolerance": threshold,
                    },
                )
            )
            predictable = any(
                value is not None for value in (period, affine, polynomial_degree)
            )
            claims.append(
                claim(
                    kind="procedural_predictability",
                    status="supported" if predictable else "rejected",
                    exact=observation.exhaustive
                    and (
                        not predictable
                        or any(
                            value is not None
                            for value in (
                                exact_period,
                                exact_affine,
                                exact_polynomial_degree,
                            )
                        )
                    ),
                    facts={
                        **base,
                        "periodic": period is not None,
                        "affine_recurrence": affine is not None,
                        "polynomial": polynomial_degree is not None,
                    },
                )
            )
            details.append(
                {
                    "tensor": parameter.tensor_name,
                    "period": period,
                    "exact_period": exact_period,
                    "affine_recurrence": affine,
                    "exact_affine_recurrence": exact_affine,
                    "polynomial_degree": polynomial_degree,
                    "exact_polynomial_degree": exact_polynomial_degree,
                }
            )
        return AnalyzerResult(claims=tuple(claims), details={"tensors": details})


def _procedural_facts(
    values: np.ndarray,
    threshold: float,
) -> dict:
    return {
        "period": _smallest_period(values, threshold),
        "exact_period": _smallest_period(values, 0.0),
        "affine_recurrence": _affine_recurrence(values, threshold),
        "exact_affine_recurrence": _affine_recurrence(values, 0.0),
        "polynomial_degree": _polynomial_degree(values, threshold),
        "exact_polynomial_degree": _polynomial_degree(values, 0.0),
    }


def _smallest_period(values: np.ndarray, tolerance: float) -> int | None:
    if values.size < 2:
        return 1
    maximum = min(values.size // 2, 256)
    for period in range(1, maximum + 1):
        if np.all(np.abs(values[period:] - values[:-period]) <= tolerance):
            return period
    return None


def _affine_recurrence(
    values: np.ndarray,
    tolerance: float,
) -> dict[str, float] | None:
    if values.size < 3:
        return None
    x = values[:-1]
    y = values[1:]
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    centered_x = x - x_mean
    denominator = float(centered_x @ centered_x)
    if denominator > np.finfo(np.float64).eps * max(1.0, float(x @ x)):
        scale = float(centered_x @ (y - y_mean)) / denominator
        offset = y_mean - scale * x_mean
    else:
        design = np.column_stack((x, np.ones_like(x)))
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        scale = float(coefficients[0])
        offset = float(coefficients[1])
    predicted = x * scale + offset
    error = float(np.max(np.abs(predicted - y)))
    if error > tolerance:
        return None
    return {
        "scale": scale,
        "offset": offset,
        "maximum_error": error,
    }


def _polynomial_degree(values: np.ndarray, tolerance: float) -> int | None:
    if values.size < 2:
        return 0
    differences = values
    for degree in range(0, min(6, values.size - 1) + 1):
        if float(np.max(differences) - np.min(differences)) <= tolerance:
            return degree
        differences = np.diff(differences)
    return None
