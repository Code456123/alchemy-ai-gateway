"""In-memory pricing cache."""

from __future__ import annotations

from datetime import datetime

from loguru import logger


class PricingCache:
    """In-memory cache for model pricing data."""

    def __init__(self) -> None:
        self.data: dict[str, dict[str, dict]] = {}

    def get(self, provider: str, model: str) -> dict | None:
        """Returns {input_price, output_price, currency, last_updated} or None."""
        if provider in self.data and model in self.data[provider]:
            return self.data[provider][model]
        return None

    def set(self, provider: str, model: str, pricing: dict) -> None:
        """Stores pricing with timestamp."""
        if provider not in self.data:
            self.data[provider] = {}
        pricing["last_updated"] = datetime.utcnow().isoformat()
        self.data[provider][model] = pricing
        logger.debug(
            "Cached pricing for {}:{} - input=${:.6f}, output=${:.6f}",
            provider,
            model,
            pricing.get("input_price", 0),
            pricing.get("output_price", 0),
        )

    def refresh(self) -> None:
        """Clear all cached prices for manual refresh."""
        self.data.clear()
        logger.info("Pricing cache cleared for refresh")

    def get_all(self, provider: str) -> dict[str, dict] | None:
        """Returns all cached models for a provider."""
        return self.data.get(provider)
