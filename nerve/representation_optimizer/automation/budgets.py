from __future__ import annotations

from dataclasses import replace

from nerve.representation_optimizer.automation.contracts import (
    BudgetAdmission,
    BudgetUsage,
    CandidateResourceCost,
    OptimizationBudget,
)
from nerve.representation_optimizer.providers.types import ProviderCandidatePlan


class BudgetLedger:
    def __init__(self, budget: OptimizationBudget) -> None:
        self._budget = budget
        self._usage = BudgetUsage()

    @property
    def usage(self) -> BudgetUsage:
        return self._usage

    def admit_scope(self) -> tuple[bool, str]:
        maximum = self._budget.maximum_scopes
        if maximum is not None and self._usage.scopes >= maximum:
            return False, "whole scope skipped: maximum_scopes is exhausted"
        self._usage = replace(self._usage, scopes=self._usage.scopes + 1)
        return True, "whole scope admitted"

    def admit_candidate(
        self,
        plan: ProviderCandidatePlan,
        *,
        execution_nanoseconds: int | None,
    ) -> BudgetAdmission:
        estimate = plan.static_estimate
        cost = CandidateResourceCost(
            permanent_bytes=estimate.permanent_bytes,
            transient_bytes=estimate.transient_bytes,
            construction_nanoseconds=estimate.construction_nanoseconds,
            execution_nanoseconds=execution_nanoseconds,
            # sanity + matched benchmark + full-local + whole-model
            experiment_invocations=4,
        )
        before = self._usage
        reasons = []
        if not estimate.feasible:
            reasons.extend(estimate.reasons)
        proposed = BudgetUsage(
            scopes=before.scopes,
            candidates=before.candidates + 1,
            permanent_bytes=before.permanent_bytes + cost.permanent_bytes,
            # Candidates execute sequentially, so transient memory is a peak,
            # not a cumulative allocation across the run.
            transient_bytes=max(before.transient_bytes, cost.transient_bytes),
            construction_nanoseconds=(
                before.construction_nanoseconds
                + (cost.construction_nanoseconds or 0)
            ),
            execution_nanoseconds=(
                before.execution_nanoseconds
                + (cost.execution_nanoseconds or 0)
            ),
            experiment_invocations=(
                before.experiment_invocations + cost.experiment_invocations
            ),
        )
        checks = (
            ("maximum_candidates", proposed.candidates),
            ("maximum_permanent_bytes", proposed.permanent_bytes),
            ("maximum_transient_bytes", proposed.transient_bytes),
            (
                "maximum_experiment_invocations",
                proposed.experiment_invocations,
            ),
        )
        for field, proposed_value in checks:
            maximum = getattr(self._budget, field)
            if maximum is not None and proposed_value > maximum:
                reasons.append(
                    f"whole candidate exceeds {field}: "
                    f"{proposed_value} > {maximum}"
                )
        construction_limit = self._budget.maximum_construction_nanoseconds
        if construction_limit is not None:
            if cost.construction_nanoseconds is None:
                reasons.append(
                    "whole candidate has no calibrated construction-cost "
                    "estimate required by maximum_construction_nanoseconds"
                )
            elif proposed.construction_nanoseconds > construction_limit:
                reasons.append(
                    "whole candidate exceeds maximum_construction_nanoseconds: "
                    f"{proposed.construction_nanoseconds} > {construction_limit}"
                )
        execution_limit = self._budget.maximum_execution_nanoseconds
        if execution_limit is not None:
            if cost.execution_nanoseconds is None:
                reasons.append(
                    "whole candidate has no target execution-cost estimate "
                    "required by maximum_execution_nanoseconds"
                )
            elif proposed.execution_nanoseconds > execution_limit:
                reasons.append(
                    "whole candidate exceeds maximum_execution_nanoseconds: "
                    f"{proposed.execution_nanoseconds} > {execution_limit}"
                )
        admitted = not reasons
        if admitted:
            reasons.append(
                "whole candidate admitted; construction and every benchmark "
                "and validation phase will run at declared quality"
            )
            self._usage = proposed
        return BudgetAdmission(
            admitted=admitted,
            reasons=tuple(reasons),
            cost=cost,
            usage_before=before,
            usage_after=self._usage,
        )
