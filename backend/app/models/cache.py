"""Semantic cache data models — cache entries, verification results, and decisions."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.app.constants.models import ModelID


class CacheMetadata(BaseModel):
    """Extracted metadata stored alongside each cache entry for verification."""

    model_config = ConfigDict(frozen=True)

    intent: str = Field(description="Normalized user intent extracted from the query.")
    entities: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Key entities extracted from the query.",
    )
    relationships: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Entity-relationship triples as 'entity|relation|target' strings.",
    )


class CacheEntry(BaseModel):
    """A single cached query-response pair with verification metadata."""

    model_config = ConfigDict(frozen=True)

    entry_id: str = Field(description="Unique identifier for this cache entry.")
    query: str = Field(description="The original user query.")
    response_text: str = Field(description="The cached response text.")
    embedding: list[float] = Field(description="Query embedding vector.")
    metadata: CacheMetadata = Field(description="Extracted verification metadata.")
    model_used: ModelID | None = Field(default=None)
    cost_usd: float = Field(ge=0.0, default=0.0)
    latency_ms: float = Field(ge=0.0, default=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ttl_hours: int = Field(gt=0, default=168)

    @property
    def is_expired(self) -> bool:
        """True if the entry has exceeded its TTL."""
        age_hours = (datetime.now(UTC) - self.created_at).total_seconds() / 3600
        return age_hours > self.ttl_hours


class GateResult(BaseModel):
    """Result from a single verification gate."""

    model_config = ConfigDict(frozen=True)

    gate_name: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str


class VerificationResult(BaseModel):
    """Aggregated result from the 3-gate verification pipeline."""

    model_config = ConfigDict(frozen=True)

    intent_result: GateResult
    entity_result: GateResult
    relationship_result: GateResult
    embedding_similarity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    is_cache_hit: bool

    def explain(self) -> str:
        """Human-readable summary of the verification decision."""
        status = "HIT" if self.is_cache_hit else "MISS"
        return (
            f"Cache {status} (confidence={self.confidence:.2f}): "
            f"intent={self.intent_result.score:.2f}, "
            f"entity={self.entity_result.score:.2f}, "
            f"relationship={self.relationship_result.score:.2f}, "
            f"embedding={self.embedding_similarity:.2f}"
        )


class CacheDecision(BaseModel):
    """Final cache decision returned to the pipeline."""

    model_config = ConfigDict(frozen=True)

    is_hit: bool
    entry: CacheEntry | None = None
    verification: VerificationResult | None = None
    lookup_latency_ms: float = Field(ge=0.0, default=0.0)
