"""Working Memory — fast in-memory store for the active session.

Layer 1 of the two-layer memory architecture. Stores recent conversation
chunks for the current session with sub-millisecond retrieval.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime

from loguru import logger

from backend.app.context.models import SemanticChunk


class WorkingMemory:
    """In-memory ring buffer of semantic chunks for the current session."""

    def __init__(self, max_chunks: int = 50) -> None:
        self._chunks: deque[SemanticChunk] = deque(maxlen=max_chunks)
        self._max_chunks = max_chunks

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[SemanticChunk]:
        return list(self._chunks)

    def add(self, chunk: SemanticChunk) -> None:
        """Add a chunk to working memory."""
        self._chunks.append(chunk)
        logger.trace("Working memory: added chunk {}, size={}", chunk.chunk_id, self.size)

    def add_many(self, chunks: list[SemanticChunk]) -> None:
        for chunk in chunks:
            self.add(chunk)

    def get_recent(self, n: int = 10) -> list[SemanticChunk]:
        """Get the N most recent chunks."""
        items = list(self._chunks)
        return items[-n:] if len(items) > n else items

    def search_by_topic(self, topic: str) -> list[SemanticChunk]:
        """Find chunks matching a topic keyword."""
        topic_lower = topic.lower()
        return [c for c in self._chunks if topic_lower in c.topic.lower() or topic_lower in c.text.lower()]

    def get_session_chunks(self, session_id: str) -> list[SemanticChunk]:
        """Get all chunks for a specific session."""
        return [c for c in self._chunks if c.session_id == session_id]

    def total_tokens(self) -> int:
        """Total token count across all chunks."""
        return sum(c.token_count for c in self._chunks)

    def get_topics(self) -> list[str]:
        """Get unique topics from working memory."""
        topics: list[str] = []
        seen: set[str] = set()
        for c in self._chunks:
            if c.topic and c.topic not in seen:
                topics.append(c.topic)
                seen.add(c.topic)
        return topics

    def clear(self) -> None:
        self._chunks.clear()
        logger.debug("Working memory cleared")
