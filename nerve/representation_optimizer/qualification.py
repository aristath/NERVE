from __future__ import annotations

from dataclasses import dataclass

from nerve.compilation import Json, ModelCompileError


QUALIFICATION_REGIME_SCHEMA = "nerve.optimizer.qualification_regime.v1"


@dataclass(frozen=True)
class QualificationRegime:
    """Runtime conditions under which product behavior is qualified."""

    speculative_draft_tokens: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.speculative_draft_tokens, bool)
            or not isinstance(self.speculative_draft_tokens, int)
            or self.speculative_draft_tokens < 0
        ):
            raise ModelCompileError(
                "qualification speculative draft tokens must be a "
                "non-negative integer"
            )

    def to_json(self) -> Json:
        return {
            "schema": QUALIFICATION_REGIME_SCHEMA,
            "speculative_draft_tokens": self.speculative_draft_tokens,
        }

    @classmethod
    def from_json(cls, document: Json) -> QualificationRegime:
        if (
            set(document)
            != {"schema", "speculative_draft_tokens"}
            or document.get("schema") != QUALIFICATION_REGIME_SCHEMA
        ):
            raise ModelCompileError(
                "optimizer qualification regime is invalid"
            )
        return cls(
            speculative_draft_tokens=document["speculative_draft_tokens"],
        )
