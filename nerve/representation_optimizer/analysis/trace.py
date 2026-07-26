from __future__ import annotations

import numpy as np

from nerve.representation_optimizer.analysis.claims import AnalyzerResult, claim
from nerve.representation_optimizer.analysis.context import ScopeAnalysisContext


class ReachableActivationAnalyzer:
    analyzer_id = "reachable_activation_evidence"
    version = "1"

    def analyze(self, context: ScopeAnalysisContext) -> AnalyzerResult:
        trace = context.activation_trace
        if trace is None:
            return AnalyzerResult(
                claims=(
                    claim(
                        kind="reachable_activation_refinement",
                        status="inconclusive",
                        exact=False,
                        facts={"reason": "no activation trace was supplied"},
                    ),
                ),
                details={"trace": None},
            )
        claims = []
        signals = []
        for signal_id, raw_values in sorted(trace.signals.items()):
            values = np.asarray(raw_values, dtype=np.float64)
            flat = values.reshape(-1)
            finite = bool(np.all(np.isfinite(flat)))
            zero_ratio = (
                float(np.count_nonzero(flat == 0) / flat.size) if flat.size else 1.0
            )
            numerical_rank = None
            if values.ndim >= 2:
                matrix = values.reshape(-1, values.shape[-1])
                if max(matrix.shape) <= context.budget.decomposition_dimension_limit:
                    numerical_rank = int(np.linalg.matrix_rank(matrix))
            facts = {
                "signal_id": signal_id,
                "trace_domain": trace.domain,
                "trace_digest": trace.trace_digest,
                "shape": list(values.shape),
                "finite": finite,
                "observed_zero_ratio": zero_ratio,
                "observed_numerical_rank": numerical_rank,
                "sampled_behavior_is_exhaustive": False,
            }
            claims.append(
                claim(
                    kind="reachable_activation_refinement",
                    status="supported" if finite else "rejected",
                    exact=False,
                    facts=facts,
                )
            )
            signals.append(facts)
        return AnalyzerResult(
            claims=tuple(claims),
            details={
                "trace_domain": trace.domain,
                "trace_digest": trace.trace_digest,
                "signals": signals,
            },
        )
