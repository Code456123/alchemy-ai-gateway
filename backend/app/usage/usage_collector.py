"""Usage collection from responses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from loguru import logger


@dataclass
class RuntimeUsage:
    """Runtime usage information from a single request."""

    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    timestamp: str  # ISO 8601
    cache_hit: bool = False
    economic_mode: bool = False
    manual_override: bool = False
    success: bool = True

    def total_tokens(self) -> int:
        """Return total tokens (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens


class UsageCollector:
    """Collects runtime usage data from responses."""

    @staticmethod
    def collect(
        response: dict,
        provider: str,
        model: str,
        latency_ms: float = 0.0,
        cache_hit: bool = False,
        economic_mode: bool = False,
        manual_override: bool = False,
        success: bool = True,
    ) -> RuntimeUsage:
        """
        Collect usage data from response.

        Args:
            response: Response dict with usage info
            provider: Provider name
            model: Model name
            latency_ms: Request latency in milliseconds
            cache_hit: Whether response came from cache
            economic_mode: Whether economic mode is active
            manual_override: Whether manual model override was used
            success: Whether request succeeded

        Returns:
            RuntimeUsage object
        """
        prompt_tokens = response.get("usage", {}).get("prompt_tokens", 0)
        completion_tokens = response.get("usage", {}).get("completion_tokens", 0)

        usage = RuntimeUsage(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow().isoformat(),
            cache_hit=cache_hit,
            economic_mode=economic_mode,
            manual_override=manual_override,
            success=success,
        )

        logger.debug(
            "Usage collected: {}:{} tokens={} latency={}ms cache_hit={}",
            provider,
            model,
            usage.total_tokens(),
            latency_ms,
            cache_hit,
        )

        return usage
