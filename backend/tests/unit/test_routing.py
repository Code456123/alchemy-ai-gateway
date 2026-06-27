"""Unit tests for the rule-based routing engine with decision score."""

from __future__ import annotations

import pytest

from backend.app.constants.enums import (
    RoutingAction,
    SecurityStatus,
    TaskType,
    ThreatType,
)
from backend.app.constants.models import ModelID
from backend.app.models.analysis import FastDetectorResult, PromptAnalysis, SecurityResult
from backend.app.models.budget import BudgetSnapshot
from backend.app.routing import RoutingEngine


@pytest.fixture
def engine() -> RoutingEngine:
    return RoutingEngine()


@pytest.fixture
def healthy_budget() -> BudgetSnapshot:
    return BudgetSnapshot(daily_limit_usd=5.0, spent_usd=0.0)


@pytest.fixture
def clear() -> SecurityResult:
    return SecurityResult(status=SecurityStatus.CLEAR, reason="ok")


def test_security_block_short_circuits(
    engine: RoutingEngine, healthy_budget: BudgetSnapshot
) -> None:
    blocked = SecurityResult(
        status=SecurityStatus.BLOCK, threats=(ThreatType.INJECTION,), reason="bad"
    )
    decision = engine.decide(security=blocked, analysis=None, budget=healthy_budget)
    assert decision.action is RoutingAction.BLOCK
    assert decision.model is None


def test_fast_path_routes_to_groq_13b(
    engine: RoutingEngine, clear: SecurityResult, healthy_budget: BudgetSnapshot
) -> None:
    fast = FastDetectorResult(is_fast_path=True, reason="greeting")
    decision = engine.decide(
        security=clear, analysis=None, budget=healthy_budget, fast_detector=fast
    )
    assert decision.action is RoutingAction.MODEL_CALL
    assert decision.model is ModelID.GROQ_LLAMA2_13B


def test_critical_budget_forces_groq_13b(engine: RoutingEngine, clear: SecurityResult) -> None:
    broke = BudgetSnapshot(daily_limit_usd=5.0, spent_usd=5.0)
    analysis = PromptAnalysis(
        task_type=TaskType.CODING, complexity=0.9, needs_coding=True, reason="x"
    )
    decision = engine.decide(security=clear, analysis=analysis, budget=broke)
    assert decision.model is ModelID.GROQ_LLAMA2_13B


def test_high_complexity_routes_to_groq_70b(
    engine: RoutingEngine, clear: SecurityResult, healthy_budget: BudgetSnapshot
) -> None:
    analysis = PromptAnalysis(
        task_type=TaskType.REASONING,
        complexity=0.9,
        needs_reasoning=True,
        needs_coding=True,
        reason="x",
    )
    decision = engine.decide(security=clear, analysis=analysis, budget=healthy_budget)
    assert decision.model is ModelID.GROQ_LLAMA2_70B
    assert decision.score_breakdown is not None
    assert decision.score_breakdown.band == "high"


def test_low_complexity_routes_to_groq_13b(
    engine: RoutingEngine, clear: SecurityResult, healthy_budget: BudgetSnapshot
) -> None:
    analysis = PromptAnalysis(task_type=TaskType.GENERAL, complexity=0.1, reason="x")
    decision = engine.decide(security=clear, analysis=analysis, budget=healthy_budget)
    assert decision.model is ModelID.GROQ_LLAMA2_13B
    assert decision.score_breakdown is not None
    assert decision.score_breakdown.band == "low"


def test_vision_triggers_hard_gate(
    engine: RoutingEngine, clear: SecurityResult, healthy_budget: BudgetSnapshot
) -> None:
    analysis = PromptAnalysis(
        task_type=TaskType.GENERAL, complexity=0.2, needs_vision=True, reason="x"
    )
    decision = engine.decide(security=clear, analysis=analysis, budget=healthy_budget)
    assert decision.model is ModelID.GROQ_MIXTRAL
    assert decision.score_breakdown is not None
    assert decision.score_breakdown.hard_gate == "vision_unsupported"


def test_override_is_respected(
    engine: RoutingEngine, clear: SecurityResult, healthy_budget: BudgetSnapshot
) -> None:
    analysis = PromptAnalysis(task_type=TaskType.GENERAL, complexity=0.1, reason="x")
    decision = engine.decide(
        security=clear, analysis=analysis, budget=healthy_budget, model_override="gpt4o"
    )
    assert decision.model is ModelID.GPT4O


def test_medium_complexity_routes_to_mixtral(
    engine: RoutingEngine, clear: SecurityResult, healthy_budget: BudgetSnapshot
) -> None:
    analysis = PromptAnalysis(task_type=TaskType.QA, complexity=0.5, reason="x")
    decision = engine.decide(security=clear, analysis=analysis, budget=healthy_budget)
    assert decision.model is ModelID.GROQ_MIXTRAL
    assert decision.score_breakdown is not None
    assert decision.score_breakdown.band == "mid"
    assert ModelID.GROQ_LLAMA2_70B in decision.fallback_chain


def test_score_breakdown_explain(
    engine: RoutingEngine, clear: SecurityResult, healthy_budget: BudgetSnapshot
) -> None:
    analysis = PromptAnalysis(
        task_type=TaskType.CODING,
        complexity=0.7,
        needs_coding=True,
        needs_reasoning=True,
        reason="x",
    )
    decision = engine.decide(security=clear, analysis=analysis, budget=healthy_budget)
    assert decision.score_breakdown is not None
    explanation = decision.score_breakdown.explain()
    assert "Score" in explanation
    assert "complexity=" in explanation


def test_economic_mode_applies_penalty(
    engine: RoutingEngine, clear: SecurityResult, healthy_budget: BudgetSnapshot
) -> None:
    analysis = PromptAnalysis(
        task_type=TaskType.CODING, complexity=0.5, needs_coding=True, reason="x"
    )
    normal = engine.decide(
        security=clear, analysis=analysis, budget=healthy_budget, economic_mode=False
    )
    economic = engine.decide(
        security=clear, analysis=analysis, budget=healthy_budget, economic_mode=True
    )
    assert economic.score_breakdown is not None
    assert economic.score_breakdown.economic_penalty == -5.0
    assert economic.score_breakdown.total_score < normal.score_breakdown.total_score


def test_groq_model_aliases(
    engine: RoutingEngine, clear: SecurityResult, healthy_budget: BudgetSnapshot
) -> None:
    analysis = PromptAnalysis(task_type=TaskType.GENERAL, complexity=0.1, reason="x")
    for alias, expected in [
        ("mixtral", ModelID.GROQ_MIXTRAL),
        ("groq_70b", ModelID.GROQ_LLAMA2_70B),
        ("groq_13b", ModelID.GROQ_LLAMA2_13B),
    ]:
        decision = engine.decide(
            security=clear, analysis=analysis, budget=healthy_budget, model_override=alias
        )
        assert decision.model is expected, f"alias '{alias}' should map to {expected}"


def test_no_analysis_routes_to_mixtral(
    engine: RoutingEngine, clear: SecurityResult, healthy_budget: BudgetSnapshot
) -> None:
    decision = engine.decide(security=clear, analysis=None, budget=healthy_budget)
    assert decision.model is ModelID.GROQ_MIXTRAL
