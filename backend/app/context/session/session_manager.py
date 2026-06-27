"""Session Manager — handles session lifecycle, summary, and restoration."""

from __future__ import annotations

from uuid import uuid4

from loguru import logger

from backend.app.context.memory.unified_memory_service import UnifiedMemoryService
from backend.app.context.memory.working_memory import WorkingMemory
from backend.app.context.models import SemanticChunk, SessionSummary


class SessionManager:
    """Manages session lifecycle including creation, summary, and restoration."""

    def __init__(
        self,
        unified_memory: UnifiedMemoryService,
        working_memory: WorkingMemory,
    ) -> None:
        self._unified = unified_memory
        self._working = working_memory
        self._current_session_id: str = ""
        self._previous_summary: SessionSummary | None = None

    @property
    def current_session_id(self) -> str:
        return self._current_session_id

    @property
    def previous_summary(self) -> SessionSummary | None:
        return self._previous_summary

    def start_session(self, session_id: str | None = None) -> str:
        """Start a new session, optionally restoring from a previous one."""
        self._current_session_id = session_id or uuid4().hex[:12]
        logger.info("Session started: {}", self._current_session_id)
        return self._current_session_id

    def end_session(self) -> SessionSummary | None:
        """End the current session: generate summary, store to unified memory, clear working."""
        if not self._current_session_id:
            return None

        chunks = self._working.chunks
        if not chunks:
            logger.info("Session {} ended with no conversation", self._current_session_id)
            self._working.clear()
            return None

        self._unified.store_chunks(chunks)

        summary = self._unified.generate_session_summary(
            self._current_session_id,
            chunks,
        )
        self._unified.store_session_summary(summary)

        logger.info(
            "Session {} ended: {} chunks stored, summary generated",
            self._current_session_id,
            len(chunks),
        )

        self._working.clear()
        self._previous_summary = summary
        return summary

    def restore_session(self, session_id: str) -> SessionSummary | None:
        """Restore context from a previous session into working memory."""
        chunks = self._unified.restore_session(session_id)
        if not chunks:
            logger.info("No chunks found for session {}", session_id)
            return None

        summaries = [c for c in chunks if c.topic == "session_summary"]
        if summaries:
            summary_chunk = summaries[-1]
            self._previous_summary = SessionSummary(
                session_id=session_id,
                project_info=summary_chunk.text,
            )
        else:
            self._previous_summary = self._unified.generate_session_summary(session_id, chunks)

        conversation_chunks = [c for c in chunks if c.topic != "session_summary"]
        self._working.add_many(conversation_chunks)

        logger.info(
            "Restored session {}: {} chunks loaded",
            session_id,
            len(conversation_chunks),
        )
        return self._previous_summary
