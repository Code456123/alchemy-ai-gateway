"""Prompt Builder — assembles the final prompt from context + query.

Final Prompt = System Prompt + Optimized Context + Current User Query

Independent of any specific LLM gateway.
"""

from __future__ import annotations

import time

from loguru import logger

from backend.app.context.models import ContextResult, RankedChunk, SemanticChunk, SessionSummary

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Use the conversation context provided "
    "below to give accurate, relevant responses. If previous context is "
    "available, maintain continuity with prior discussions."
)

_CHARS_PER_TOKEN = 4


class PromptBuilder:
    """Builds the final prompt sent to the LLM gateway."""

    def __init__(self, system_prompt: str = _DEFAULT_SYSTEM_PROMPT) -> None:
        self._system_prompt = system_prompt

    def build(
        self,
        user_query: str,
        ranked_chunks: list[RankedChunk] | None = None,
        session_summary: SessionSummary | None = None,
        max_context_tokens: int = 2000,
    ) -> ContextResult:
        """Build the final prompt with optimized context."""
        start = time.perf_counter()

        context_parts: list[str] = []
        tokens_used = 0
        chunks_used = 0

        if session_summary:
            summary_text = f"[Previous Session Context]\n{session_summary.to_text()}\n"
            summary_tokens = len(summary_text) // _CHARS_PER_TOKEN
            if tokens_used + summary_tokens <= max_context_tokens:
                context_parts.append(summary_text)
                tokens_used += summary_tokens

        if ranked_chunks:
            context_parts.append("[Relevant Context]")
            for rc in ranked_chunks:
                chunk_tokens = rc.chunk.token_count or (len(rc.chunk.text) // _CHARS_PER_TOKEN)
                if tokens_used + chunk_tokens > max_context_tokens:
                    break
                speaker = rc.chunk.speaker.value.capitalize()
                context_parts.append(f"{speaker}: {rc.chunk.text}")
                tokens_used += chunk_tokens
                chunks_used += 1

        context_text = "\n".join(context_parts) if context_parts else ""

        total_tokens = (
            len(self._system_prompt) // _CHARS_PER_TOKEN
            + tokens_used
            + len(user_query) // _CHARS_PER_TOKEN
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            "Prompt built: {} context chunks, ~{} tokens in {:.1f}ms",
            chunks_used,
            total_tokens,
            elapsed_ms,
        )

        return ContextResult(
            system_prompt=self._system_prompt,
            context_text=context_text,
            user_query=user_query,
            chunks_used=chunks_used,
            chunks_retrieved=len(ranked_chunks) if ranked_chunks else 0,
            total_context_tokens=total_tokens,
            strategy_used="chunks" if ranked_chunks else "none",
            session_restored=session_summary is not None,
        )

    def build_compressed(
        self,
        user_query: str,
        chunks: list[SemanticChunk],
        max_tokens: int = 500,
    ) -> ContextResult:
        """Build a compressed prompt for critical budget state.

        Concatenates chunk summaries (or truncated text) to fit within budget.
        """
        context_parts: list[str] = ["[Compressed Context]"]
        tokens_used = 0

        for chunk in chunks:
            text = chunk.summary if chunk.summary else chunk.text
            if len(text) > max_tokens * _CHARS_PER_TOKEN:
                text = text[: max_tokens * _CHARS_PER_TOKEN] + "..."
            chunk_tokens = len(text) // _CHARS_PER_TOKEN
            if tokens_used + chunk_tokens > max_tokens:
                break
            context_parts.append(text)
            tokens_used += chunk_tokens

        return ContextResult(
            system_prompt=self._system_prompt,
            context_text="\n".join(context_parts),
            user_query=user_query,
            chunks_used=len(context_parts) - 1,
            total_context_tokens=tokens_used,
            strategy_used="summary",
        )
