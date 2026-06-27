"""Google Gemini pricing adapter."""

from __future__ import annotations

from loguru import logger


class GeminiAdapter:
    """Google Gemini model pricing adapter."""

    def __init__(self) -> None:
        self.pricing: dict[str, dict] = {
            "gemini-pro": {
                "input_price": 0.000125,
                "output_price": 0.000375,
                "currency": "USD",
                "unit": "per_1k_tokens",
            },
            "gemini-1.5-pro": {
                "input_price": 0.0075,
                "output_price": 0.03,
                "currency": "USD",
                "unit": "per_1m_tokens",
            },
            "gemini-1.5-flash": {
                "input_price": 0.075,
                "output_price": 0.30,
                "currency": "USD",
                "unit": "per_1m_tokens",
            },
        }

    async def get_model_pricing(self, model_name: str) -> dict:
        """Returns pricing info for Gemini model."""
        normalized = model_name.lower().strip()

        for key in self.pricing:
            if key in normalized:
                pricing = self.pricing[key].copy()
                pricing["source"] = "cache"
                logger.debug("Gemini pricing for {}: ${:.6f} in, ${:.6f} out", model_name, pricing["input_price"], pricing["output_price"])
                return pricing

        logger.warning("Unknown Gemini model: {}", model_name)
        raise ValueError(f"Unknown Gemini model: {model_name}")
