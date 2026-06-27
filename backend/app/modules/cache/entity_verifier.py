"""Gate 2.2 — Entity Verifier.

Compares the key entities extracted from the current query against those in a
cached entry. Uses set overlap (Jaccard-like) to produce a [0, 1] score.
"""

from __future__ import annotations

from backend.app.models.cache import CacheMetadata, GateResult

# Minimum overlap ratio for the gate to pass.
_ENTITY_PASS_THRESHOLD = 0.6


class EntityVerifier:
    """Verifies whether two queries reference the same key entities."""

    def __init__(self, pass_threshold: float = _ENTITY_PASS_THRESHOLD) -> None:
        self._threshold = pass_threshold

    def verify(self, current: CacheMetadata, stored: CacheMetadata) -> GateResult:
        """Compare entity sets between the current query and a cached entry.

        Args:
            current: Metadata extracted from the incoming query.
            stored: Metadata stored with the cache entry.

        Returns:
            A :class:`GateResult` with a [0, 1] overlap score.
        """
        current_set = set(current.entities)
        stored_set = set(stored.entities)

        if not current_set and not stored_set:
            return GateResult(
                gate_name="entity",
                passed=True,
                score=1.0,
                reason="Both queries have no entities",
            )

        if not current_set or not stored_set:
            return GateResult(
                gate_name="entity",
                passed=False,
                score=0.0,
                reason="One query has entities, the other does not",
            )

        intersection = current_set & stored_set
        union = current_set | stored_set
        score = len(intersection) / len(union) if union else 0.0

        passed = score >= self._threshold
        return GateResult(
            gate_name="entity",
            passed=passed,
            score=round(score, 4),
            reason=(
                f"Entity overlap {len(intersection)}/{len(union)} "
                f"({score:.2f}): {intersection or 'none'}"
            ),
        )
