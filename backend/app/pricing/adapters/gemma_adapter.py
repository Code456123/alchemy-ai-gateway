"""Google Gemma pricing adapter."""

from __future__ import annotations

from loguru import logger


class GemmaAdapter:
    """Google Gemma model pricing adapter."""

    def __init__(self) -> None:
        self.pricing: dict[str, dict] = {
            "gemma-7b": {
                "input_price": 0.0001,
                "output_price": 0.0003,
                "currency": "USD",
                "unit": "per_1k_tokens",
            },
            "gemma-2b": {
                "input_price": 0.00005,
                "output_price": 0.00015,
                "currency": "USD",
                "unit": "per_1k_tokens",
            },
        }

    async def get_model_pricing(self, model_name: str) -> dict:
        """Returns pricing info for Gemma model."""
        normalized = model_name.lower().strip()

        for key in self.pricing:
            if key in normalized:
                pricing = self.pricing[key].copy()
                pricing["source"] = "cache"
                logger.debug(
                    "Gemma pricing for {}: ${:.6f} in, ${:.6f} out",
                    model_name,
                    pricing["input_price"],
                    pricing["output_price"],
                )
                return pricing

        logger.warning("Unknown Gemma model: {}", model_name)
        raise ValueError(f"Unknown Gemma model: {model_name}")
