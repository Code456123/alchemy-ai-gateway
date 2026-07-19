"""Mock provider pricing adapter."""

from __future__ import annotations

from loguru import logger

from backend.app.constants.models import MODEL_COSTS


class MockAdapter:
    """Pricing adapter for mock responses."""

    async def get_model_pricing(self, model_name: str) -> dict:
        normalized = model_name.lower().strip()
        for key, pricing in MODEL_COSTS.items():
            if key == normalized or key.value == normalized:
                result = {
                    "input_price": pricing.get("input", 0.0),
                    "output_price": pricing.get("output", 0.0),
                    "currency": "USD",
                    "unit": "per_1k_tokens",
                    "source": "cache",
                }
                logger.debug("Mock pricing for {}: ${:.6f} in, ${:.6f} out", model_name, result["input_price"], result["output_price"])
                return result

        logger.warning("Unknown mock model: %s", model_name)
        raise ValueError(f"Unknown mock model: {model_name}")
