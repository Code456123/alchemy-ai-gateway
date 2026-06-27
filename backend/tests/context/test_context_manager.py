"""Integration tests for the full ContextManager."""

import pytest

from backend.app.context import ContextManager
from backend.app.models.budget import BudgetSnapshot


class TestContextManager:
    def setup_method(self):
        self.cm = ContextManager()

    def test_session_created(self):
        assert self.cm.session_id != ""

    def test_prepare_context_empty(self):
        result = self.cm.prepare_context("What is Python?")
        assert result.user_query == "What is Python?"
        assert result.system_prompt != ""

    def test_update_and_retrieve(self):
        self.cm.update_memory(
            user_query="What is Python?",
            response_text="Python is a programming language.",
            model_used="gpt4o",
            topic="coding",
        )
        assert self.cm.working_memory.size >= 2

        result = self.cm.prepare_context("Tell me more about Python")
        assert result.chunks_used >= 0

    def test_budget_aware_context(self):
        self.cm.update_memory("Q1", "A1", "gpt4o", "test")
        self.cm.update_memory("Q2", "A2", "gpt4o", "test")

        healthy = BudgetSnapshot(daily_limit_usd=5.0, spent_usd=0.1)
        result_healthy = self.cm.prepare_context("Query", budget=healthy)

        critical = BudgetSnapshot(
            daily_limit_usd=5.0,
            spent_usd=4.9,
            warning_threshold=0.75,
            critical_threshold=0.90,
        )
        result_critical = self.cm.prepare_context("Query", budget=critical)
        assert result_critical.strategy_used in ("summary", "chunks")

    def test_end_session(self):
        self.cm.update_memory("Hello", "Hi there", "local_2b")
        summary = self.cm.end_session()
        assert summary is not None
        assert summary.total_queries >= 1

    def test_cross_model_context(self):
        self.cm.update_memory("Explain ML", "ML is...", "gpt4o", "ml")
        result = self.cm.get_cross_model_context("More about ML", "claude-sonnet")
        assert result.user_query == "More about ML"
