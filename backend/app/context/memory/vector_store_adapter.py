"""Vector store adapter — abstracts the vector DB behind a common interface.

Supports two backends:
- ``LocalVectorStoreAdapter``: in-memory brute-force search (zero dependencies)
- ``PineconeAdapter``: Pinecone managed vector DB (persistent, cross-session)

The ``create_vector_adapter`` factory picks the right backend based on settings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from loguru import logger

from backend.app.storage.vector_store import VectorStore


@dataclass
class VectorSearchResult:
    """Result from vector store search."""

    chunk_id: str
    similarity: float
    metadata: dict[str, object]


class VectorStoreAdapter(ABC):
    """Abstract interface for vector storage backends."""

    @abstractmethod
    def upsert(
        self, chunk_id: str, embedding: list[float], metadata: dict[str, object]
    ) -> None: ...

    @abstractmethod
    def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        filter_dict: dict[str, object] | None = None,
    ) -> list[VectorSearchResult]: ...

    @abstractmethod
    def delete(self, chunk_id: str) -> None: ...

    @abstractmethod
    def delete_by_session(self, session_id: str) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...


class LocalVectorStoreAdapter(VectorStoreAdapter):
    """Local in-memory adapter using the existing VectorStore."""

    def __init__(self, dimension: int) -> None:
        self._store = VectorStore(dimension)
        self._metadata: dict[str, dict[str, object]] = {}

    def upsert(self, chunk_id: str, embedding: list[float], metadata: dict[str, object]) -> None:
        self._store.add(chunk_id, embedding)
        self._metadata[chunk_id] = metadata
        logger.trace("Local store: upserted chunk {}", chunk_id)

    def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        filter_dict: dict[str, object] | None = None,
    ) -> list[VectorSearchResult]:
        results = self._store.search(embedding, top_k=top_k * 2)
        out: list[VectorSearchResult] = []
        for r in results:
            meta = self._metadata.get(r.entry_id, {})
            if filter_dict and not all(meta.get(k) == v for k, v in filter_dict.items()):
                continue
            out.append(
                VectorSearchResult(chunk_id=r.entry_id, similarity=r.similarity, metadata=meta)
            )
            if len(out) >= top_k:
                break
        return out

    def delete(self, chunk_id: str) -> None:
        self._store.remove(chunk_id)
        self._metadata.pop(chunk_id, None)

    def delete_by_session(self, session_id: str) -> None:
        to_remove = [
            cid for cid, meta in self._metadata.items() if meta.get("session_id") == session_id
        ]
        for cid in to_remove:
            self.delete(cid)

    def clear(self) -> None:
        self._store.clear()
        self._metadata.clear()


# ── Pinecone Adapter ────────────────────────────────────────


_PINECONE_TEXT_LIMIT = 1000


class PineconeAdapter(VectorStoreAdapter):
    """Pinecone vector store adapter (SDK v9+).

    Metadata schema per vector:
        chunk_id, session_id, text (truncated 1000 chars), timestamp,
        token_count, topic, summary, importance_score, chunk_type,
        speaker, model_used.

    Index requirements:
        metric: cosine
        dimension: must match the embedding engine (512 for BoW)
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        namespace: str = "alchemy",
    ) -> None:
        try:
            from pinecone import Pinecone
        except ImportError as exc:
            raise ImportError("pinecone package is required: uv pip install pinecone") from exc

        self._namespace = namespace
        self._pc = Pinecone(api_key=api_key)
        self._index = self._pc.Index(index_name)

        logger.info(
            "Pinecone adapter connected index={} namespace={}",
            index_name,
            namespace,
        )

    def upsert(self, chunk_id: str, embedding: list[float], metadata: dict[str, object]) -> None:
        meta = {k: v for k, v in metadata.items() if v is not None}
        if "text" in meta:
            text = str(meta["text"])
            if len(text) > _PINECONE_TEXT_LIMIT:
                meta["text"] = text[:_PINECONE_TEXT_LIMIT]

        self._index.upsert(
            vectors=[(chunk_id, embedding, meta)],
            namespace=self._namespace,
        )
        logger.debug("Pinecone upsert chunk_id={}", chunk_id)

    def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        filter_dict: dict[str, object] | None = None,
    ) -> list[VectorSearchResult]:
        response = self._index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=self._namespace,
            filter=filter_dict or None,
        )

        results: list[VectorSearchResult] = []
        for match in response.matches:
            results.append(
                VectorSearchResult(
                    chunk_id=match.id,
                    similarity=match.score,
                    metadata=dict(match.metadata) if match.metadata else {},
                )
            )

        logger.debug("Pinecone query returned {} results (top_k={})", len(results), top_k)
        return results

    def delete(self, chunk_id: str) -> None:
        self._index.delete(ids=[chunk_id], namespace=self._namespace)
        logger.debug("Pinecone delete chunk_id={}", chunk_id)

    def delete_by_session(self, session_id: str) -> None:
        self._index.delete(
            filter={"session_id": session_id},
            namespace=self._namespace,
        )
        logger.debug("Pinecone delete_by_session session_id={}", session_id)

    def clear(self) -> None:
        self._index.delete(delete_all=True, namespace=self._namespace)
        logger.info("Pinecone namespace '{}' cleared", self._namespace)


# ── Factory ─────────────────────────────────────────────────


def create_vector_adapter(
    *,
    pinecone_api_key: str = "",
    pinecone_index_name: str = "",
    pinecone_namespace: str = "alchemy",
    dimension: int = 512,
) -> VectorStoreAdapter:
    """Return PineconeAdapter if credentials are set, else LocalVectorStoreAdapter.

    This is the single entry point for adapter construction. The ContextManager
    calls this in its ``__init__``.
    """
    if pinecone_api_key and pinecone_index_name:
        logger.info("Vector store: Pinecone (index={})", pinecone_index_name)
        return PineconeAdapter(
            api_key=pinecone_api_key,
            index_name=pinecone_index_name,
            namespace=pinecone_namespace,
        )

    logger.info("Vector store: local in-memory (no PINECONE_API_KEY)")
    return LocalVectorStoreAdapter(dimension=dimension)
