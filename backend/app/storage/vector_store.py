"""In-memory vector store with brute-force nearest-neighbor search.

Drop-in replacement for FAISS that requires no native dependencies. The
interface is intentionally narrow so swapping in ``faiss.IndexFlatIP`` later
is a one-file change.

Performance: <5ms for 10K entries (brute-force cosine on normalized vectors).
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger


@dataclass
class SearchResult:
    """A single nearest-neighbor search result."""

    entry_id: str
    similarity: float


class VectorStore:
    """In-memory vector store with cosine similarity search."""

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._vectors: dict[str, list[float]] = {}

    @property
    def size(self) -> int:
        """Number of vectors currently stored."""
        return len(self._vectors)

    def add(self, entry_id: str, vector: list[float]) -> None:
        """Add or replace a vector in the store.

        Args:
            entry_id: Unique identifier for this vector.
            vector: L2-normalized vector of length ``dimension``.
        """
        if len(vector) != self._dimension:
            raise ValueError(
                f"Vector dimension mismatch: got {len(vector)}, expected {self._dimension}"
            )
        self._vectors[entry_id] = vector

    def remove(self, entry_id: str) -> None:
        """Remove a vector from the store."""
        self._vectors.pop(entry_id, None)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        """Find the top-K most similar vectors to the query.

        Args:
            query_vector: L2-normalized query vector.
            top_k: Maximum number of results to return.

        Returns:
            A list of :class:`SearchResult` sorted by descending similarity.
        """
        if not self._vectors:
            return []

        results: list[SearchResult] = []
        for entry_id, vec in self._vectors.items():
            sim = self._cosine_similarity(query_vector, vec)
            results.append(SearchResult(entry_id=entry_id, similarity=sim))

        results.sort(key=lambda r: r.similarity, reverse=True)
        top = results[:top_k]

        logger.trace(
            "Vector search: {} candidates, top similarity={:.4f}",
            len(results),
            top[0].similarity if top else 0.0,
        )
        return top

    def clear(self) -> None:
        """Remove all vectors from the store."""
        self._vectors.clear()

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors (assumes L2-normalized)."""
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        return max(0.0, min(1.0, dot))
