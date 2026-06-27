"""Anthropic pricing adapter."""

from __future__ import annotations

from loguru import logger


class AnthropicAdapter:
    """Anthropic model pricing adapter."""

    def __init__(self) -> None:
        self.pricing: dict[str, dict] = {
            "claude-sonnet": {
                "input_price": 0.003,
                "output_price": 0.015,
                "currency": "USD",
                "unit": "per_1k_tokens",
            },
            "claude-opus": {
                "input_price": 0.015,
                "output_price": 0.075,
                "currency": "USD",
                "unit": "per_1k_tokens",
            },
            "claude-haiku": {
                "input_price": 0.00025,
                "output_price": 0.00125,
                "currency": "USD",
                "unit": "per_1k_tokens",
            },
        }

    async def get_model_pricing(self, model_name: str) -> dict:
        """Returns pricing info for Anthropic model."""
        normalized = model_name.lower().strip()

        for key in self.pricing:
            if key in normalized:
                pricing = self.pricing[key].copy()
                pricing["source"] = "cache"
                logger.debug(
                    "Anthropic pricing for {}: ${:.6f} in, ${:.6f} out",
                    model_name,
                    pricing["input_price"],
                    pricing["output_price"],
                )
                return pricing

        logger.warning("Unknown Anthropic model: {}", model_name)
        raise ValueError(f"Unknown Anthropic model: {model_name}")
