"""Unit tests for PromptBuilder."""

import pytest

from backend.app.context.prompt_builder import PromptBuilder
from backend.app.context.models import RankedChunk, SemanticChunk, SessionSummary, Speaker


class TestPromptBuilder:
    def setup_method(self):
        self.builder = PromptBuilder()

    def test_build_with_no_context(self):
        result = self.builder.build("What is Python?")
        assert result.user_query == "What is Python?"
        assert result.chunks_used == 0
        assert result.system_prompt != ""

    def test_build_with_chunks(self):
        chunks = [
            RankedChunk(
                chunk=SemanticChunk(text="Python is a language", token_count=5, speaker=Speaker.ASSISTANT),
                relevance_score=0.8,
            ),
        ]
        result = self.builder.build("What is Python?", ranked_chunks=chunks)
        assert result.chunks_used == 1
        assert "Python is a language" in result.context_text

    def test_build_with_session_summary(self):
        summary = SessionSummary(
            session_id="s1",
            project_info="QueryWise AI project",
            topics_discussed=["routing", "budget"],
        )
        result = self.builder.build("Continue our work", session_summary=summary)
        assert result.session_restored is True
        assert "QueryWise AI" in result.context_text

    def test_build_compressed(self):
        chunks = [
            SemanticChunk(text="Some long text about Python", token_count=50, speaker=Speaker.ASSISTANT),
        ]
        result = self.builder.build_compressed("Query", chunks, max_tokens=100)
        assert result.strategy_used == "summary"

    def test_token_budget_respected(self):
        chunks = [
            RankedChunk(
                chunk=SemanticChunk(text="x" * 10000, token_count=2500, speaker=Speaker.ASSISTANT),
                relevance_score=0.9,
            ),
        ]
        result = self.builder.build("Query", ranked_chunks=chunks, max_context_tokens=100)
        assert result.chunks_used == 0
