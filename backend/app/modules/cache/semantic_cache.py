"""Semantic Cache — 3-gate verified response cache.

Orchestrates the full cache lookup flow:

    Query → Embedding → FAISS Search (Gate 1)
        → Structural Verification (Gate 2: intent + entity + relationship)
        → Confidence Engine (Gate 3)
        → Cache Hit / Miss

The cache NEVER trusts embedding similarity alone. A cached response is only
reused when it passes all three verification gates and the weighted confidence
score exceeds the threshold.
"""

from __future__ import annotations

import time
import uuid

from loguru import logger

from backend.app.config.settings import Settings, get_settings
from backend.app.constants.models import ModelID
from backend.app.embeddings import EmbeddingEngine
from backend.app.models.cache import CacheDecision, CacheEntry
from backend.app.modules.cache.confidence_engine import ConfidenceEngine
from backend.app.modules.cache.entity_verifier import EntityVerifier
from backend.app.modules.cache.intent_verifier import IntentVerifier
from backend.app.modules.cache.metadata_extractor import MetadataExtractor
from backend.app.modules.cache.relationship_verifier import RelationshipVerifier
from backend.app.storage import VectorStore

# Gate 1 retrieves this many candidates before verification.
_DEFAULT_TOP_K = 3


class SemanticCache:
    """3-gate verified semantic cache for query-response pairs.

    Thread-safe for single-writer, multi-reader use (the typical CLI pattern).
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        embedding_engine: EmbeddingEngine | None = None,
        confidence_threshold: float | None = None,
        top_k: int = _DEFAULT_TOP_K,
    ) -> None:
        self._settings = settings or get_settings()
        self._embedding = embedding_engine or EmbeddingEngine()
        self._store = VectorStore(dimension=self._embedding.dimension)
        self._entries: dict[str, CacheEntry] = {}
        self._extractor = MetadataExtractor()
        self._intent_verifier = IntentVerifier()
        self._entity_verifier = EntityVerifier()
        self._relationship_verifier = RelationshipVerifier()
        self._confidence = ConfidenceEngine(
            threshold=(
                confidence_threshold
                if confidence_threshold is not None
                else self._settings.cache_similarity_threshold
            ),
        )
        self._top_k = top_k

    @property
    def size(self) -> int:
        """Number of entries currently in the cache."""
        return len(self._entries)

    @property
    def confidence_threshold(self) -> float:
        """The minimum confidence required for a cache hit."""
        return self._confidence.threshold

    def lookup(self, query: str) -> CacheDecision:
        """Look up a query in the cache using the 3-gate verification pipeline.

        Args:
            query: The raw user query.

        Returns:
            A :class:`CacheDecision` indicating hit/miss with full trace.
        """
        start = time.perf_counter()

        if not self._entries:
            return CacheDecision(
                is_hit=False,
                lookup_latency_ms=self._elapsed_ms(start),
            )

        # ── Gate 1: Candidate Retrieval ──
        query_embedding = self._embedding.encode(query)
        candidates = self._store.search(query_embedding, top_k=self._top_k)

        if not candidates:
            logger.debug("Cache MISS: no candidates found")
            return CacheDecision(
                is_hit=False,
                lookup_latency_ms=self._elapsed_ms(start),
            )

        # Extract metadata for the current query.
        current_metadata = self._extractor.extract(query)

        # ── Gate 2 + 3: Verify each candidate ──
        for candidate in candidates:
            entry = self._entries.get(candidate.entry_id)
            if entry is None or entry.is_expired:
                if entry is not None and entry.is_expired:
                    self._evict(candidate.entry_id)
                continue

            # Gate 2: Structural verification (intent + entity + relationship).
            intent_result = self._intent_verifier.verify(current_metadata, entry.metadata)
            entity_result = self._entity_verifier.verify(current_metadata, entry.metadata)
            relationship_result = self._relationship_verifier.verify(
                current_metadata, entry.metadata
            )

            # Gate 3: Confidence engine.
            verification = self._confidence.evaluate(
                intent=intent_result,
                entity=entity_result,
                relationship=relationship_result,
                embedding_similarity=candidate.similarity,
            )

            if verification.is_cache_hit:
                latency = self._elapsed_ms(start)
                logger.info(
                    "Cache HIT entry_id={} confidence={:.4f} latency={:.2f}ms: {}",
                    candidate.entry_id,
                    verification.confidence,
                    latency,
                    verification.explain(),
                )
                return CacheDecision(
                    is_hit=True,
                    entry=entry,
                    verification=verification,
                    lookup_latency_ms=latency,
                )

            logger.debug(
                "Cache candidate rejected entry_id={}: {}",
                candidate.entry_id,
                verification.explain(),
            )

        latency = self._elapsed_ms(start)
        logger.debug("Cache MISS: no candidate passed verification (latency={:.2f}ms)", latency)
        return CacheDecision(
            is_hit=False,
            lookup_latency_ms=latency,
        )

    def store(
        self,
        query: str,
        response_text: str,
        *,
        model_used: ModelID | None = None,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
    ) -> CacheEntry:
        """Store a query-response pair in the cache.

        Args:
            query: The raw user query.
            response_text: The model's response.
            model_used: Which model produced the response.
            cost_usd: Cost of the model call.
            latency_ms: Latency of the model call.

        Returns:
            The created :class:`CacheEntry`.
        """
        entry_id = uuid.uuid4().hex[:12]
        embedding = self._embedding.encode(query)
        metadata = self._extractor.extract(query)

        entry = CacheEntry(
            entry_id=entry_id,
            query=query,
            response_text=response_text,
            embedding=embedding,
            metadata=metadata,
            model_used=model_used,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            ttl_hours=self._settings.cache_default_ttl_hours,
        )

        self._entries[entry_id] = entry
        self._store.add(entry_id, embedding)

        logger.info(
            "Cache STORE entry_id={} intent={} entities={} relationships={}",
            entry_id,
            metadata.intent,
            metadata.entities,
            metadata.relationships,
        )
        return entry

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._entries.clear()
        self._store.clear()

    def _evict(self, entry_id: str) -> None:
        """Remove a single entry (e.g. expired) from both stores."""
        self._entries.pop(entry_id, None)
        self._store.remove(entry_id)
        logger.debug("Cache EVICT entry_id={}", entry_id)

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        """Milliseconds elapsed since ``start``."""
        return round((time.perf_counter() - start) * 1000, 3)
