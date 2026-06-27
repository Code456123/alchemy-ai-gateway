"""Gate 2.1 — Intent Verifier.

Compares the normalized intent of the current query against a cached entry's
intent. Returns a binary match score (1.0 = same intent, 0.0 = different).
"""

from __future__ import annotations

from backend.app.models.cache import CacheMetadata, GateResult


class IntentVerifier:
    """Verifies whether two queries share the same user intent."""

    def verify(self, current: CacheMetadata, stored: CacheMetadata) -> GateResult:
        """Compare the intent of the current query against a cached entry.

        Args:
            current: Metadata extracted from the incoming query.
            stored: Metadata stored with the cache entry.

        Returns:
            A :class:`GateResult` with score 1.0 (match) or 0.0 (mismatch).
        """
        if current.intent == stored.intent:
            return GateResult(
                gate_name="intent",
                passed=True,
                score=1.0,
                reason=f"Intent match: {current.intent}",
            )
        return GateResult(
            gate_name="intent",
            passed=False,
            score=0.0,
            reason=f"Intent mismatch: '{current.intent}' vs '{stored.intent}'",
        )
