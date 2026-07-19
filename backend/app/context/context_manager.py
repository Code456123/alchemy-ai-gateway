"""Adaptive Unified Memory Context Manager.

Orchestrates the full context preparation workflow:

    User Query → Semantic Cache → Cache Miss → Context Manager → LLM → Update Memory

Core responsibilities:
- Working Memory (Layer 1): fast in-session retrieval
- Unified Memory (Layer 2): persistent cross-session semantic memory
- Semantic chunking of conversation
- Budget-aware context optimization
- Cross-model unified memory (model-independent)
- Session summary and restoration
"""

from __future__ import annotations

import time
from uuid import uuid4

from loguru import logger

from backend.app.config.settings import Settings, get_settings
from backend.app.constants.enums import BudgetState
from backend.app.context.chunking import SemanticChunker
from backend.app.context.memory import UnifiedMemoryService, WorkingMemory
from backend.app.context.memory.vector_store_adapter import (
    VectorStoreAdapter,
    create_vector_adapter,
)
from backend.app.constants.enums import TaskType
from backend.app.context.models import (
    ChunkType,
    ContextResult,
    SemanticChunk,
    SessionSummary,
    Speaker,
)
from backend.app.context.prompt_builder import PromptBuilder
from backend.app.context.retrieval import ContextRelevanceFilter
from backend.app.context.session import SessionManager
from backend.app.embeddings.engine import EmbeddingEngine
from backend.app.models.budget import BudgetSnapshot


class ContextManager:
    """Adaptive Unified Memory Context Manager.

    Executed ONLY after the Semantic Cache returns a Cache Miss.
    Optimizes token usage, latency, and context quality without
    sacrificing response accuracy.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        embedding_engine: EmbeddingEngine | None = None,
        vector_adapter: VectorStoreAdapter | None = None,
        working_memory: WorkingMemory | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embedding = embedding_engine or EmbeddingEngine()

        adapter = vector_adapter or create_vector_adapter(
            pinecone_api_key=self._settings.pinecone_api_key,
            pinecone_index_name=self._settings.pinecone_index_name,
            pinecone_namespace=self._settings.pinecone_namespace,
            dimension=self._embedding.dimension,
        )
        self._working_memory = working_memory or WorkingMemory(max_chunks=50)
        self._unified_memory = UnifiedMemoryService(adapter, self._embedding)
        self._chunker = SemanticChunker()
        self._relevance_filter = ContextRelevanceFilter()
        self._prompt_builder = PromptBuilder()
        self._session_manager = SessionManager(self._unified_memory, self._working_memory)

        self._session_id = self._session_manager.start_session()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def working_memory(self) -> WorkingMemory:
        return self._working_memory

    def prepare_context(
        self,
        user_query: str,
        budget: BudgetSnapshot | None = None,
        top_k: int = 10,
        task_type: TaskType | None = None,
    ) -> ContextResult:
        """Prepare optimized context for an LLM call.

        This is the main entry point called after a semantic cache miss.

        Steps:
        1. Search Working Memory first (fast, <5ms)
        2. If insufficient, search Unified Memory via vector store
        3. Apply relevance filter to rank candidates
        4. Build prompt based on budget state
        """
        start = time.perf_counter()

        max_chunks = self._max_chunks_for_budget(budget)

        candidates: list[tuple[SemanticChunk, float]] = []

        wm_chunks = self._working_memory.search_by_topic(user_query)
        if not wm_chunks:
            wm_chunks = self._working_memory.get_recent(max_chunks)

        for chunk in wm_chunks:
            sim = self._embedding.similarity(
                self._embedding.encode(user_query),
                self._embedding.encode(chunk.text),
            )
            candidates.append((chunk, sim))

        if len(candidates) < top_k:
            um_results = self._unified_memory.retrieve_chunks(
                user_query, top_k=top_k, session_id=None
            )
            existing_ids = {c.chunk_id for c, _ in candidates}
            for chunk, sim in um_results:
                if chunk.chunk_id not in existing_ids:
                    candidates.append((chunk, sim))

        ranked = self._relevance_filter.rank(
            candidates,
            max_chunks=max_chunks,
            current_session_id=self._session_id,
            task_type=task_type,
        )

        selected_chunks = [
            r for r in ranked if r.similarity_score >= 0.75
        ]

        if selected_chunks:
            for ranked_chunk in selected_chunks:
                preview = ranked_chunk.chunk.text.replace("\n", " ")[:80]
                logger.debug(
                    "Selected context chunk: similarity={:.3f}, preview='{}'",
                    ranked_chunk.similarity_score,
                    preview,
                )
        else:
            logger.debug(
                "No context chunks selected: all candidates below similarity threshold 0.75"
            )

        strategy = self._budget_strategy(budget)
        if strategy == "compressed":
            result = self._prompt_builder.build_compressed(
                user_query,
                [r.chunk for r in selected_chunks],
                max_tokens=self._settings.context_summary_max_tokens,
            )
        else:
            result = self._prompt_builder.build(
                user_query,
                ranked_chunks=selected_chunks,
                session_summary=self._session_manager.previous_summary,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Context prepared in {:.1f}ms: {} chunks, ~{} tokens, strategy={}",
            elapsed_ms,
            result.chunks_used,
            result.total_context_tokens,
            result.strategy_used,
        )
        return result

    def update_memory(
        self,
        user_query: str,
        response_text: str,
        model_used: str = "",
        topic: str = "",
    ) -> None:
        """Update memory after a successful LLM response.

        Chunks both the user query and assistant response, stores in working memory.
        Unified memory is updated at session end.
        """
        user_chunks = self._chunker.chunk_message(
            text=user_query,
            session_id=self._session_id,
            speaker=Speaker.USER,
            chunk_type=ChunkType.USER_QUERY,
            model_used=model_used,
            topic=topic,
        )
        response_chunks = self._chunker.chunk_message(
            text=response_text,
            session_id=self._session_id,
            speaker=Speaker.ASSISTANT,
            chunk_type=ChunkType.MODEL_RESPONSE,
            model_used=model_used,
            topic=topic,
        )

        for chunk in user_chunks + response_chunks:
            chunk.embedding = self._embedding.encode(chunk.text)
            self._working_memory.add(chunk)

        logger.debug(
            "Memory updated: {} user chunks + {} response chunks",
            len(user_chunks),
            len(response_chunks),
        )

    def end_session(self) -> SessionSummary | None:
        """End the current session and persist to unified memory."""
        return self._session_manager.end_session()

    def restore_session(self, session_id: str) -> SessionSummary | None:
        """Restore context from a previous session."""
        summary = self._session_manager.restore_session(session_id)
        if summary:
            logger.info("Session {} restored", session_id)
        return summary

    def get_cross_model_context(
        self,
        query: str,
        new_model: str,
        top_k: int = 5,
    ) -> ContextResult:
        """Retrieve relevant context when switching models mid-session.

        The unified memory is model-independent, so any model can access
        conversation history from any other model.
        """
        logger.info("Cross-model context transfer → {}", new_model)
        return self.prepare_context(query, top_k=top_k)

    def _max_chunks_for_budget(self, budget: BudgetSnapshot | None) -> int:
        """Determine max chunks based on budget state."""
        if budget is None:
            return self._settings.context_max_chunks_healthy

        if budget.state == BudgetState.CRITICAL:
            return 1
        if budget.state == BudgetState.LOW:
            return self._settings.context_max_chunks_low
        return self._settings.context_max_chunks_healthy

    @staticmethod
    def _budget_strategy(budget: BudgetSnapshot | None) -> str:
        """Determine context strategy based on budget state."""
        if budget is None:
            return "full"
        if budget.state == BudgetState.CRITICAL:
            return "compressed"
        if budget.state == BudgetState.LOW:
            return "reduced"
        return "full"
