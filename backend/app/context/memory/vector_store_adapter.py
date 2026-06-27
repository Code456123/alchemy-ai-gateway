"""Vector store adapter — abstracts Pinecone behind a common interface.

This is the only file that needs to change when replacing the vector DB backend.
Currently uses the in-memory VectorStore as a local fallback; Pinecone integration
will be added when the API key is available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from loguru import logger

from backend.app.context.models import SemanticChunk
from backend.app.storage.vector_store import VectorStore


@dataclass
class VectorSearchResult:
    """Result from vector store search."""

    chunk_id: str
    similarity: float
    metadata: dict


class VectorStoreAdapter(ABC):
    """Abstract interface for vector storage backends."""

    @abstractmethod
    def upsert(self, chunk_id: str, embedding: list[float], metadata: dict) -> None:
        ...

    @abstractmethod
    def query(self, embedding: list[float], top_k: int = 10, filter_dict: dict | None = None) -> list[VectorSearchResult]:
        ...

    @abstractmethod
    def delete(self, chunk_id: str) -> None:
        ...

    @abstractmethod
    def delete_by_session(self, session_id: str) -> None:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...


class LocalVectorStoreAdapter(VectorStoreAdapter):
    """Local in-memory adapter using the existing VectorStore."""

    def __init__(self, dimension: int) -> None:
        self._store = VectorStore(dimension)
        self._metadata: dict[str, dict] = {}

    def upsert(self, chunk_id: str, embedding: list[float], metadata: dict) -> None:
        self._store.add(chunk_id, embedding)
        self._metadata[chunk_id] = metadata
        logger.trace("Local store: upserted chunk {}", chunk_id)

    def query(self, embedding: list[float], top_k: int = 10, filter_dict: dict | None = None) -> list[VectorSearchResult]:
        results = self._store.search(embedding, top_k=top_k * 2)
        out: list[VectorSearchResult] = []
        for r in results:
            meta = self._metadata.get(r.entry_id, {})
            if filter_dict:
                if not all(meta.get(k) == v for k, v in filter_dict.items()):
                    continue
            out.append(VectorSearchResult(
                chunk_id=r.entry_id,
                similarity=r.similarity,
                metadata=meta,
            ))
            if len(out) >= top_k:
                break
        return out

    def delete(self, chunk_id: str) -> None:
        self._store.remove(chunk_id)
        self._metadata.pop(chunk_id, None)

    def delete_by_session(self, session_id: str) -> None:
        to_remove = [cid for cid, meta in self._metadata.items() if meta.get("session_id") == session_id]
        for cid in to_remove:
            self.delete(cid)

    def clear(self) -> None:
        self._store.clear()
        self._metadata.clear()


class PineconeAdapter(VectorStoreAdapter):
    """Pinecone vector store adapter.

    Schema per vector:
        id: str (chunk_id)
        values: list[float] (embedding vector)
        metadata:
            chunk_id: str
            session_id: str
            text: str (truncated to 1000 chars for metadata limit)
            timestamp: str (ISO 8601)
            token_count: int
            topic: str
            summary: str
            importance_score: float
            chunk_type: str (user_query | model_response | session_summary | system)
            speaker: str (user | assistant | system)
            model_used: str

    Index configuration:
        metric: cosine
        dimension: configurable (512 for BoW, 384 for MiniLM, 768 for BERT)
        pod_type: s1 (starter)
    """

    def __init__(self, api_key: str, index_name: str, environment: str, dimension: int) -> None:
        self._api_key = api_key
        self._index_name = index_name
        self._environment = environment
        self._dimension = dimension
        self._index = None
        logger.info("Pinecone adapter initialized (index={}, dim={})", index_name, dimension)

    def _get_index(self):
        if self._index is None:
            raise NotImplementedError(
                "Pinecone API integration pending. Provide PINECONE_API_KEY and "
                "PINECONE_INDEX_NAME in .env to activate."
            )
        return self._index

    def upsert(self, chunk_id: str, embedding: list[float], metadata: dict) -> None:
        meta = {k: v for k, v in metadata.items() if v is not None}
        if "text" in meta and len(meta["text"]) > 1000:
            meta["text"] = meta["text"][:1000]
        self._get_index().upsert(vectors=[(chunk_id, embedding, meta)])

    def query(self, embedding: list[float], top_k: int = 10, filter_dict: dict | None = None) -> list[VectorSearchResult]:
        response = self._get_index().query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict,
        )
        return [
            VectorSearchResult(
                chunk_id=match["id"],
                similarity=match["score"],
                metadata=match.get("metadata", {}),
            )
            for match in response["matches"]
        ]

    def delete(self, chunk_id: str) -> None:
        self._get_index().delete(ids=[chunk_id])

    def delete_by_session(self, session_id: str) -> None:
        self._get_index().delete(filter={"session_id": session_id})

    def clear(self) -> None:
        self._get_index().delete(delete_all=True)
