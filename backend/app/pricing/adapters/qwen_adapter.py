"""Alibaba Qwen pricing adapter."""

from __future__ import annotations

from loguru import logger


class QwenAdapter:
    """Alibaba Qwen model pricing adapter."""

    def __init__(self) -> None:
        self.pricing: dict[str, dict] = {
            "qwen-plus": {
                "input_price": 0.0008,
                "output_price": 0.002,
                "currency": "USD",
                "unit": "per_1k_tokens",
            },
            "qwen-turbo": {
                "input_price": 0.0004,
                "output_price": 0.0012,
                "currency": "USD",
                "unit": "per_1k_tokens",
            },
            "qwen-max": {
                "input_price": 0.001,
                "output_price": 0.003,
                "currency": "USD",
                "unit": "per_1k_tokens",
            },
        }

    async def get_model_pricing(self, model_name: str) -> dict:
        """Returns pricing info for Qwen model."""
        normalized = model_name.lower().strip()

        for key in self.pricing:
            if key in normalized:
                pricing = self.pricing[key].copy()
                pricing["source"] = "cache"
                logger.debug("Qwen pricing for {}: ${:.6f} in, ${:.6f} out", model_name, pricing["input_price"], pricing["output_price"])
                return pricing

        logger.warning("Unknown Qwen model: {}", model_name)
        raise ValueError(f"Unknown Qwen model: {model_name}")
