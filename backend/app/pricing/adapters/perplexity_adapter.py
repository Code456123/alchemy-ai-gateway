"""Perplexity pricing adapter."""

from __future__ import annotations

from loguru import logger


class PerplexityAdapter:
    """Perplexity model pricing adapter."""

    def __init__(self) -> None:
        self.pricing: dict[str, dict] = {
            "pplx-7b-online": {
                "input_price": 0.002,
                "output_price": 0.002,
                "currency": "USD",
                "unit": "per_1k_tokens",
            },
            "pplx-70b-online": {
                "input_price": 0.01,
                "output_price": 0.01,
                "currency": "USD",
                "unit": "per_1k_tokens",
            },
        }

    async def get_model_pricing(self, model_name: str) -> dict:
        """Returns pricing info for Perplexity model."""
        normalized = model_name.lower().strip()

        for key in self.pricing:
            if key in normalized:
                pricing = self.pricing[key].copy()
                pricing["source"] = "cache"
                logger.debug("Perplexity pricing for {}: ${:.6f} in, ${:.6f} out", model_name, pricing["input_price"], pricing["output_price"])
                return pricing

        logger.warning("Unknown Perplexity model: {}", model_name)
        raise ValueError(f"Unknown Perplexity model: {model_name}")
