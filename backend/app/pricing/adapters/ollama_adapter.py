"""Ollama (local) pricing adapter."""

from __future__ import annotations

from loguru import logger


class OllamaAdapter:
    """Ollama local model pricing adapter (free)."""

    def __init__(self) -> None:
        self.pricing: dict[str, dict] = {
            "gemma:2b": {
                "input_price": 0.0,
                "output_price": 0.0,
                "currency": "USD",
                "unit": "per_1k_tokens",
            },
            "llama2": {
                "input_price": 0.0,
                "output_price": 0.0,
                "currency": "USD",
                "unit": "per_1k_tokens",
            },
            "mistral": {
                "input_price": 0.0,
                "output_price": 0.0,
                "currency": "USD",
                "unit": "per_1k_tokens",
            },
        }

    async def get_model_pricing(self, model_name: str) -> dict:
        """Returns pricing info for Ollama model (always free)."""
        logger.debug("Ollama model {} pricing: FREE (local)", model_name)
        return {
            "input_price": 0.0,
            "output_price": 0.0,
            "currency": "USD",
            "unit": "per_1k_tokens",
            "source": "cache",
        }
