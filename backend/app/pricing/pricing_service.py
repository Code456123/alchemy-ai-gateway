"""Pricing service for cost calculation."""

from __future__ import annotations

from loguru import logger

from backend.app.pricing.pricing_cache import PricingCache
from backend.app.pricing.provider_registry import ProviderRegistry


class PricingService:
    """Calculates request costs based on provider, model, and token usage."""

    def __init__(
        self,
        provider_registry: ProviderRegistry | None = None,
        pricing_cache: PricingCache | None = None,
    ) -> None:
        self.registry = provider_registry or ProviderRegistry()
        self.cache = pricing_cache or PricingCache()

    async def calculate_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """
        Calculate cost for a request.

        Args:
            provider: Provider name (e.g., "openai")
            model: Model name (e.g., "gpt-4o-mini")
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Cost in USD (float). Returns 0.0 if pricing unavailable.

        Flow:
        1. Get adapter from registry
        2. Attempt to get pricing (live or cached)
        3. Calculate: (prompt / 1000 * input_price) + (completion / 1000 * output_price)
        4. Never crash - log warning and return 0.0 if unavailable
        """
        try:
            adapter = self.registry.get_adapter(provider)
        except ValueError:
            logger.warning("Unknown provider '{}'. Cost = $0", provider)
            return 0.0

        pricing = await self._get_pricing(adapter, provider, model)
        if pricing is None:
            logger.warning("Pricing unavailable for {}:{}", provider, model)
            return 0.0

        input_price = pricing.get("input_price", 0.0)
        output_price = pricing.get("output_price", 0.0)

        prompt_cost = (prompt_tokens / 1000.0) * input_price
        completion_cost = (completion_tokens / 1000.0) * output_price
        total_cost = round(prompt_cost + completion_cost, 6)

        logger.debug(
            "Cost calculated: {}:{} tokens={}+{} cost=${:.6f}",
            provider,
            model,
            prompt_tokens,
            completion_tokens,
            total_cost,
        )
        return total_cost

    async def _get_pricing(self, adapter, provider: str, model: str) -> dict | None:
        """Attempt to get pricing from adapter (live) or cache."""
        try:
            pricing = await adapter.get_model_pricing(model)
            if pricing:
                self.cache.set(provider, model, pricing)
                return pricing
        except Exception as e:
            logger.debug("Live pricing failed for {}:{}: {}", provider, model, e)

        cached = self.cache.get(provider, model)
        if cached:
            logger.debug("Using cached pricing for {}:{}", provider, model)
            return cached

        return None
