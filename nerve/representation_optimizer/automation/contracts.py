from __future__ import annotations

from dataclasses import asdict, dataclass

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import canonical_json_bytes


OPTIMIZER_RUN_SCHEMA = "nerve.optimizer.automated_run.v1"
OPTIMIZER_REPORT_SCHEMA = "nerve.optimizer.automated_report.v1"
OPTIMIZER_EVENT_SCHEMA = "nerve.optimizer.automated_event.v1"
OPTIMIZER_BUDGET_DECISION_SCHEMA = "nerve.optimizer.budget_decision.v1"


@dataclass(frozen=True)
class OptimizationBudget:
    """Whole-work admission limits; admitted experiments always run in full."""

    maximum_scopes: int | None
    maximum_candidates: int | None
    maximum_permanent_bytes: int | None
    maximum_transient_bytes: int | None
    maximum_construction_nanoseconds: int | None
    maximum_execution_nanoseconds: int | None
    maximum_experiment_invocations: int | None

    def __post_init__(self) -> None:
        for field, value in asdict(self).items():
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ModelCompileError(
                    f"optimizer budget {field} must be a non-negative integer or null"
                )

    @classmethod
    def explicitly_unbounded(cls) -> OptimizationBudget:
        return cls(
            maximum_scopes=None,
            maximum_candidates=None,
            maximum_permanent_bytes=None,
            maximum_transient_bytes=None,
            maximum_construction_nanoseconds=None,
            maximum_execution_nanoseconds=None,
            maximum_experiment_invocations=None,
        )

    def to_json(self) -> Json:
        document = asdict(self)
        canonical_json_bytes(document)
        return document


@dataclass(frozen=True)
class CandidateResourceCost:
    permanent_bytes: int
    transient_bytes: int
    construction_nanoseconds: int | None
    execution_nanoseconds: int | None
    experiment_invocations: int

    def __post_init__(self) -> None:
        for field in (
            "permanent_bytes",
            "transient_bytes",
            "experiment_invocations",
        ):
            value = getattr(self, field)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ModelCompileError(
                    f"candidate resource cost {field} must be non-negative"
                )
        if self.construction_nanoseconds is not None and (
            not isinstance(self.construction_nanoseconds, int)
            or isinstance(self.construction_nanoseconds, bool)
            or self.construction_nanoseconds < 0
        ):
            raise ModelCompileError(
                "candidate resource cost construction_nanoseconds must be "
                "non-negative or null"
            )
        if self.execution_nanoseconds is not None and (
            not isinstance(self.execution_nanoseconds, int)
            or isinstance(self.execution_nanoseconds, bool)
            or self.execution_nanoseconds < 0
        ):
            raise ModelCompileError(
                "candidate resource cost execution_nanoseconds must be "
                "non-negative or null"
            )

    def to_json(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class BudgetUsage:
    scopes: int = 0
    candidates: int = 0
    permanent_bytes: int = 0
    transient_bytes: int = 0
    construction_nanoseconds: int = 0
    execution_nanoseconds: int = 0
    experiment_invocations: int = 0

    def to_json(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class BudgetAdmission:
    admitted: bool
    reasons: tuple[str, ...]
    cost: CandidateResourceCost
    usage_before: BudgetUsage
    usage_after: BudgetUsage

    def __post_init__(self) -> None:
        if not self.reasons:
            raise ModelCompileError("budget admission requires an explanation")

    def to_json(self, *, candidate_id: str) -> Json:
        document = {
            "schema": OPTIMIZER_BUDGET_DECISION_SCHEMA,
            "candidate_id": candidate_id,
            "admitted": self.admitted,
            "reasons": list(self.reasons),
            "cost": self.cost.to_json(),
            "usage_before": self.usage_before.to_json(),
            "usage_after": self.usage_after.to_json(),
        }
        canonical_json_bytes(document)
        return document
