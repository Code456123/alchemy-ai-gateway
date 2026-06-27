"""Decision score breakdown for explainable routing."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ScoreBreakdown(BaseModel):
    """Detailed breakdown of the decision score for explainability."""

    model_config = ConfigDict(frozen=True)

    complexity_score: float = Field(ge=0.0, le=35.0)
    capability_score: float = Field(ge=0.0, le=30.0)
    budget_score: float = Field(ge=0.0, le=20.0)
    context_score: float = Field(ge=0.0, le=10.0)
    economic_penalty: float = Field(ge=-5.0, le=0.0)
    total_score: float = Field(ge=0.0, le=100.0)
    hard_gate: str | None = Field(default=None)

    @property
    def band(self) -> str:
        if self.total_score <= 30:
            return "low"
        if self.total_score <= 65:
            return "mid"
        return "high"

    def explain(self) -> str:
        if self.hard_gate:
            return f"Score {self.total_score:.0f}/100 (overridden: {self.hard_gate})"

        parts = []
        if self.complexity_score > 0:
            parts.append(f"complexity={self.complexity_score:.0f}/35")
        if self.capability_score > 0:
            parts.append(f"capability={self.capability_score:.0f}/30")
        if self.budget_score < 20:
            parts.append(f"budget={self.budget_score:.0f}/20")
        if self.context_score > 0:
            parts.append(f"context={self.context_score:.0f}/10")
        if self.economic_penalty < 0:
            parts.append(f"economic={self.economic_penalty:.0f}")

        detail = ", ".join(parts)
        return f"Score {self.total_score:.0f}/100 [{self.band}] ({detail})"
