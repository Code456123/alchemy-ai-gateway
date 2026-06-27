"""Rule-based routing engine with weighted decision score.

Combines security, fast-detector, task analysis, and budget state into
an explainable routing decision with transparent scoring.
"""

from __future__ import annotations

from loguru import logger

from backend.app.constants.enums import BudgetState, RoutingAction
from backend.app.constants.models import MODEL_COSTS, MODEL_FALLBACK_CHAIN, ModelID
from backend.app.constants.thresholds import (
    SCORE_BAND_LOW,
    SCORE_BAND_MID,
    SCORE_CONTEXT_TOKEN_THRESHOLD,
    SCORE_WEIGHT_BUDGET,
    SCORE_WEIGHT_CAPABILITY,
    SCORE_WEIGHT_COMPLEXITY,
    SCORE_WEIGHT_CONTEXT,
    SCORE_WEIGHT_ECONOMIC,
)
from backend.app.models.analysis import (
    FastDetectorResult,
    PromptAnalysis,
    SecurityResult,
)
from backend.app.models.budget import BudgetSnapshot
from backend.app.models.routing import RoutingDecision
from backend.app.models.scoring import ScoreBreakdown

_ESTIMATED_COMPLETION_TOKENS = 256


class RoutingEngine:
    """Selects the most appropriate model via weighted decision score."""

    def decide(
        self,
        *,
        security: SecurityResult,
        analysis: PromptAnalysis | None,
        budget: BudgetSnapshot,
        fast_detector: FastDetectorResult | None = None,
        prompt_tokens: int = 0,
        model_override: str | None = None,
        economic_mode: bool = False,
    ) -> RoutingDecision:
        """Produce a routing decision from upstream pipeline signals.

        Precedence: security block → override → fast-path → budget-constrained →
        decision score engine.
        """
        # 1. Security always wins.
        if security.is_blocked:
            return RoutingDecision(
                action=RoutingAction.BLOCK,
                model=None,
                reason=f"Blocked by security: {security.reason}",
            )

        # 2. Explicit override.
        if model_override:
            model = self._coerce_model(model_override)
            if model is not None:
                return self._model_call(model, prompt_tokens, f"Caller override → {model.value}")
            logger.warning("Unknown model_override '{}', ignoring", model_override)

        # 3. Fast path → cheapest Groq model.
        if fast_detector is not None and fast_detector.is_fast_path:
            return self._model_call(
                ModelID.GROQ_LLAMA2_13B,
                prompt_tokens,
                f"Fast path ({fast_detector.reason}) → Groq LLaMA 2 13B",
            )

        # 4. Budget exhausted → force cheapest.
        if budget.state is BudgetState.CRITICAL or budget.remaining_usd <= 0.0:
            return self._model_call(
                ModelID.GROQ_LLAMA2_13B,
                prompt_tokens,
                f"Budget {budget.state.value} → force Groq LLaMA 2 13B",
            )

        # 5. Decision score engine.
        if analysis is None:
            return self._model_call(
                ModelID.GROQ_MIXTRAL,
                prompt_tokens,
                "No analysis available → balanced Mixtral",
            )

        model, reason, breakdown = self._select_by_score(
            analysis, budget, prompt_tokens, economic_mode
        )
        return self._model_call(model, prompt_tokens, reason, score_breakdown=breakdown)

    def _compute_score(
        self,
        analysis: PromptAnalysis,
        budget: BudgetSnapshot,
        prompt_tokens: int,
        economic_mode: bool,
    ) -> ScoreBreakdown:
        complexity_score = round(analysis.complexity * SCORE_WEIGHT_COMPLEXITY, 2)

        capability_score = 0.0
        if analysis.needs_coding:
            capability_score += 10.0
        if analysis.needs_reasoning:
            capability_score += 10.0
        if analysis.needs_planning:
            capability_score += 5.0
        if analysis.needs_vision:
            capability_score += 5.0
        capability_score = min(capability_score, SCORE_WEIGHT_CAPABILITY)

        budget_score = {
            BudgetState.HEALTHY: SCORE_WEIGHT_BUDGET,
            BudgetState.LOW: SCORE_WEIGHT_BUDGET / 2,
            BudgetState.CRITICAL: 0.0,
        }.get(budget.state, SCORE_WEIGHT_BUDGET)

        context_score = 0.0
        if analysis.needs_context:
            context_score += 5.0
        if prompt_tokens > SCORE_CONTEXT_TOKEN_THRESHOLD:
            context_score += 5.0
        context_score = min(context_score, SCORE_WEIGHT_CONTEXT)

        economic_penalty = -SCORE_WEIGHT_ECONOMIC if economic_mode else 0.0

        total = (
            complexity_score + capability_score + budget_score + context_score + economic_penalty
        )
        total = round(max(0.0, min(100.0, total)), 2)

        hard_gate = None
        if analysis.needs_vision and budget.state is not BudgetState.CRITICAL:
            hard_gate = "vision_unsupported"

        return ScoreBreakdown(
            complexity_score=complexity_score,
            capability_score=capability_score,
            budget_score=budget_score,
            context_score=context_score,
            economic_penalty=economic_penalty,
            total_score=total,
            hard_gate=hard_gate,
        )

    def _select_by_score(
        self,
        analysis: PromptAnalysis,
        budget: BudgetSnapshot,
        prompt_tokens: int,
        economic_mode: bool,
    ) -> tuple[ModelID, str, ScoreBreakdown]:
        breakdown = self._compute_score(analysis, budget, prompt_tokens, economic_mode)

        if breakdown.hard_gate == "vision_unsupported":
            logger.warning("Vision required but Groq doesn't support it, using Mixtral")
            return ModelID.GROQ_MIXTRAL, breakdown.explain(), breakdown

        if breakdown.total_score <= SCORE_BAND_LOW:
            model = ModelID.GROQ_LLAMA2_13B
        elif breakdown.total_score <= SCORE_BAND_MID:
            model = ModelID.GROQ_MIXTRAL
        else:
            model = ModelID.GROQ_LLAMA2_70B

        return model, breakdown.explain(), breakdown

    def _model_call(
        self,
        model: ModelID,
        prompt_tokens: int,
        reason: str,
        score_breakdown: ScoreBreakdown | None = None,
    ) -> RoutingDecision:
        cost = self._estimate_cost(model, prompt_tokens)
        chain = tuple(ModelID(m) for m in MODEL_FALLBACK_CHAIN.get(model, []))
        logger.debug("Routing → {} (${:.5f}) :: {}", model.value, cost, reason)
        return RoutingDecision(
            action=RoutingAction.MODEL_CALL,
            model=model,
            reason=reason,
            estimated_cost_usd=cost,
            fallback_chain=chain,
            score_breakdown=score_breakdown,
        )

    @staticmethod
    def _estimate_cost(model: ModelID, prompt_tokens: int) -> float:
        costs = MODEL_COSTS.get(model, {"input": 0.0, "output": 0.0})
        prompt_cost = (prompt_tokens / 1000.0) * costs["input"]
        completion_cost = (_ESTIMATED_COMPLETION_TOKENS / 1000.0) * costs["output"]
        return round(prompt_cost + completion_cost, 6)

    @staticmethod
    def _coerce_model(value: str) -> ModelID | None:
        aliases = {
            "groq_13b": ModelID.GROQ_LLAMA2_13B,
            "groq_mixtral": ModelID.GROQ_MIXTRAL,
            "groq_70b": ModelID.GROQ_LLAMA2_70B,
            "mixtral": ModelID.GROQ_MIXTRAL,
            "llama2_13b": ModelID.GROQ_LLAMA2_13B,
            "llama2_70b": ModelID.GROQ_LLAMA2_70B,
            "local": ModelID.GROQ_LLAMA2_13B,
            "local_2b": ModelID.LOCAL_2B,
            "mini": ModelID.GPT4O_MINI,
            "gpt4o_mini": ModelID.GPT4O_MINI,
            "gpt4o": ModelID.GPT4O,
            "gpt-4o": ModelID.GPT4O,
        }
        return aliases.get(value.strip().lower())
