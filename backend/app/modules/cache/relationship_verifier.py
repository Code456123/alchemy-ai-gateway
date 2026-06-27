"""Gate 2.3 — Relationship Verifier.

Compares entity-relation-entity triples from the current query against those
stored with a cached entry. This is the innovation layer: two queries can share
entities but differ in their relationships (e.g. "cost of X" vs "maintenance
of X"), and this gate catches that.
"""

from __future__ import annotations

from backend.app.models.cache import CacheMetadata, GateResult

_RELATIONSHIP_PASS_THRESHOLD = 0.5


class RelationshipVerifier:
    """Verifies whether two queries express the same entity relationships."""

    def __init__(self, pass_threshold: float = _RELATIONSHIP_PASS_THRESHOLD) -> None:
        self._threshold = pass_threshold

    def verify(self, current: CacheMetadata, stored: CacheMetadata) -> GateResult:
        """Compare relationship triples between the current and cached query.

        Args:
            current: Metadata extracted from the incoming query.
            stored: Metadata stored with the cache entry.

        Returns:
            A :class:`GateResult` with a [0, 1] overlap score.
        """
        current_set = set(current.relationships)
        stored_set = set(stored.relationships)

        if not current_set and not stored_set:
            return GateResult(
                gate_name="relationship",
                passed=True,
                score=1.0,
                reason="Both queries have no relationships",
            )

        if not current_set or not stored_set:
            return GateResult(
                gate_name="relationship",
                passed=False,
                score=0.0,
                reason="One query has relationships, the other does not",
            )

        intersection = current_set & stored_set
        union = current_set | stored_set
        score = len(intersection) / len(union) if union else 0.0

        passed = score >= self._threshold
        return GateResult(
            gate_name="relationship",
            passed=passed,
            score=round(score, 4),
            reason=(
                f"Relationship overlap {len(intersection)}/{len(union)} "
                f"({score:.2f}): matched={intersection or 'none'}"
            ),
        )
