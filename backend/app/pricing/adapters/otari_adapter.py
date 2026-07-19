"""Otari pricing adapter."""

from __future__ import annotations

from loguru import logger


class OtariAdapter:
    """Otari pricing adapter for Groq-powered models."""

    async def get_model_pricing(self, model_name: str) -> dict:
        """Returns pricing info for Otari models (currently free)."""
        logger.debug("Otari model {} pricing: FREE", model_name)
        return {
            "input_price": 0.0,
            "output_price": 0.0,
            "currency": "USD",
            "unit": "per_1k_tokens",
            "source": "cache",
        }
