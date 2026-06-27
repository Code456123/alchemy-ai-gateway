"""Usage service for storing session usage data."""

from __future__ import annotations

from loguru import logger

from backend.app.usage.usage_collector import RuntimeUsage


class UsageService:
    """Stores runtime usage in session memory."""

    def __init__(self) -> None:
        self.requests: list[RuntimeUsage] = []
        self.total_tokens = 0
        self.total_cost = 0.0

    def add(self, usage: RuntimeUsage) -> None:
        """Store usage in session."""
        self.requests.append(usage)
        self.total_tokens += usage.total_tokens()
        logger.debug(
            "Usage stored: {} requests, {} total tokens", len(self.requests), self.total_tokens
        )

    def update_total_cost(self, cost: float) -> None:
        """Update total cost (called after pricing calculation)."""
        self.total_cost = round(self.total_cost + cost, 6)

    def get_model_history(self, model: str) -> list[RuntimeUsage]:
        """Return all uses of a model in this session."""
        return [u for u in self.requests if u.model == model]

    def get_provider_history(self, provider: str) -> list[RuntimeUsage]:
        """Return all uses of a provider in this session."""
        return [u for u in self.requests if u.provider == provider]

    def get_session_summary(self) -> dict:
        """Return session summary statistics."""
        models_used = set(u.model for u in self.requests)
        providers_used = set(u.provider for u in self.requests)
        cache_hits = sum(1 for u in self.requests if u.cache_hit)
        manual_overrides = sum(1 for u in self.requests if u.manual_override)

        return {
            "request_count": len(self.requests),
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "models_used": list(models_used),
            "providers_used": list(providers_used),
            "cache_hits": cache_hits,
            "manual_overrides": manual_overrides,
            "average_latency_ms": (
                sum(u.latency_ms for u in self.requests) / len(self.requests)
                if self.requests
                else 0.0
            ),
        }

    def get_requests(self) -> list[RuntimeUsage]:
        """Return all requests."""
        return self.requests.copy()
