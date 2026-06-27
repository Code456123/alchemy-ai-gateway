"""DeepSeek pricing adapter."""

from __future__ import annotations

from loguru import logger


class DeepSeekAdapter:
    """DeepSeek model pricing adapter."""

    def __init__(self) -> None:
        self.pricing: dict[str, dict] = {
            "deepseek-chat": {
                "input_price": 0.00014,
                "output_price": 0.00028,
                "currency": "USD",
                "unit": "per_1k_tokens",
            },
            "deepseek-coder": {
                "input_price": 0.00014,
                "output_price": 0.00028,
                "currency": "USD",
                "unit": "per_1k_tokens",
            },
        }

    async def get_model_pricing(self, model_name: str) -> dict:
        """Returns pricing info for DeepSeek model."""
        normalized = model_name.lower().strip()

        for key in self.pricing:
            if key in normalized:
                pricing = self.pricing[key].copy()
                pricing["source"] = "cache"
                logger.debug("DeepSeek pricing for {}: ${:.6f} in, ${:.6f} out", model_name, pricing["input_price"], pricing["output_price"])
                return pricing

        logger.warning("Unknown DeepSeek model: {}", model_name)
        raise ValueError(f"Unknown DeepSeek model: {model_name}")
