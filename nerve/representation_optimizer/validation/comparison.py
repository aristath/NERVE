from __future__ import annotations

from nerve.compilation import Json, ModelCompileError


def compare_exact_role_results(
    request: Json,
    reference_result: Json,
    candidate_result: Json,
    *,
    divergence_diagnostic: str,
) -> Json:
    if request["behavioral_contract"]["mode"] != "exact":
        raise ModelCompileError(
            "exact result comparator received an approximate contract"
        )
    identical = (
        reference_result["output_digest"]
        == candidate_result["output_digest"]
        and reference_result["state_digest"]
        == candidate_result["state_digest"]
    )
    return {
        "metrics": [
            {
                "name": name,
                "reference_value": 1.0,
                "candidate_value": 1.0 if identical else 0.0,
                "error": 0.0 if identical else 1.0,
                "unit": "exact_match",
            }
            for name in request["check"]["metrics"]
        ],
        "diagnostics": (
            [] if identical else [divergence_diagnostic]
        ),
    }
