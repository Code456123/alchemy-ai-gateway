"""OpenAI pricing adapter."""

from __future__ import annotations

from loguru import logger


class OpenAIAdapter:
    """OpenAI model pricing adapter."""

    def __init__(self) -> None:
        self.pricing: dict[str, dict] = {
            "gpt-4o": {
                "input_price": 0.005,
                "output_price": 0.015,
                "currency": "USD",
                "unit": "per_1k_tokens",
            },
            "gpt-4o-mini": {
                "input_price": 0.00015,
                "output_price": 0.0006,
                "currency": "USD",
                "unit": "per_1k_tokens",
            },
            "gpt-4-turbo": {
                "input_price": 0.01,
                "output_price": 0.03,
                "currency": "USD",
                "unit": "per_1k_tokens",
            },
        }

    async def get_model_pricing(self, model_name: str) -> dict:
        """
        Returns pricing info for OpenAI model.
        {
            "input_price": float,
            "output_price": float,
            "currency": "USD",
            "source": "api" | "cache",
            "last_updated": ISO 8601
        }
        """
        normalized = model_name.lower().strip()

        if normalized in self.pricing:
            pricing = self.pricing[normalized].copy()
            pricing["source"] = "cache"
            logger.debug("OpenAI pricing for {}: ${:.6f} in, ${:.6f} out", model_name, pricing["input_price"], pricing["output_price"])
            return pricing

        logger.warning("Unknown OpenAI model: {}", model_name)
        raise ValueError(f"Unknown OpenAI model: {model_name}")
