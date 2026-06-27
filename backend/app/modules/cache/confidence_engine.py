"""Gate 3 — Confidence Engine.

Combines all verification gate results and embedding similarity into a single
weighted confidence score. The cache decision is made by comparing this score
against a configurable threshold.

Weights (from spec):
    Intent match:        40%
    Relationship match:  30%
    Entity match:        20%
    Embedding similarity: 10%
"""

from __future__ import annotations

from loguru import logger

from backend.app.models.cache import GateResult, VerificationResult

# Default weights — must sum to 1.0.
_WEIGHT_INTENT = 0.40
_WEIGHT_RELATIONSHIP = 0.30
_WEIGHT_ENTITY = 0.20
_WEIGHT_EMBEDDING = 0.10

# Default confidence threshold for a cache hit.
_DEFAULT_CONFIDENCE_THRESHOLD = 0.90


class ConfidenceEngine:
    """Computes a weighted confidence score from verification gate results."""

    def __init__(self, threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        """Create the confidence engine.

        Args:
            threshold: Minimum confidence for a cache hit (0-1).
        """
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        """The minimum confidence required for a cache hit."""
        return self._threshold

    def evaluate(
        self,
        *,
        intent: GateResult,
        entity: GateResult,
        relationship: GateResult,
        embedding_similarity: float,
    ) -> VerificationResult:
        """Compute the final confidence score and cache decision.

        Args:
            intent: Result from Gate 2.1 (intent verifier).
            entity: Result from Gate 2.2 (entity verifier).
            relationship: Result from Gate 2.3 (relationship verifier).
            embedding_similarity: Raw cosine similarity from FAISS (Gate 1).

        Returns:
            A :class:`VerificationResult` with the final decision.
        """
        confidence = (
            _WEIGHT_INTENT * intent.score
            + _WEIGHT_RELATIONSHIP * relationship.score
            + _WEIGHT_ENTITY * entity.score
            + _WEIGHT_EMBEDDING * embedding_similarity
        )
        confidence = round(max(0.0, min(1.0, confidence)), 4)
        is_hit = confidence >= self._threshold

        logger.debug(
            "Confidence={:.4f} (threshold={:.2f}): intent={:.2f}*{}, "
            "rel={:.2f}*{}, entity={:.2f}*{}, embed={:.2f}*{} → {}",
            confidence,
            self._threshold,
            intent.score,
            _WEIGHT_INTENT,
            relationship.score,
            _WEIGHT_RELATIONSHIP,
            entity.score,
            _WEIGHT_ENTITY,
            embedding_similarity,
            _WEIGHT_EMBEDDING,
            "HIT" if is_hit else "MISS",
        )

        return VerificationResult(
            intent_result=intent,
            entity_result=entity,
            relationship_result=relationship,
            embedding_similarity=round(embedding_similarity, 4),
            confidence=confidence,
            is_cache_hit=is_hit,
        )
