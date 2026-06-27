"""Provider registry mapping providers to adapters."""

from __future__ import annotations

from loguru import logger

from backend.app.pricing.adapters.anthropic_adapter import AnthropicAdapter
from backend.app.pricing.adapters.deepseek_adapter import DeepSeekAdapter
from backend.app.pricing.adapters.gemini_adapter import GeminiAdapter
from backend.app.pricing.adapters.gemma_adapter import GemmaAdapter
from backend.app.pricing.adapters.grok_adapter import GrokAdapter
from backend.app.pricing.adapters.ollama_adapter import OllamaAdapter
from backend.app.pricing.adapters.openai_adapter import OpenAIAdapter
from backend.app.pricing.adapters.perplexity_adapter import PerplexityAdapter
from backend.app.pricing.adapters.qwen_adapter import QwenAdapter


class ProviderRegistry:
    """Maps provider name to adapter instance."""

    def __init__(self) -> None:
        self.adapters: dict[str, object] = {
            "openai": OpenAIAdapter(),
            "anthropic": AnthropicAdapter(),
            "gemini": GeminiAdapter(),
            "grok": GrokAdapter(),
            "qwen": QwenAdapter(),
            "gemma": GemmaAdapter(),
            "perplexity": PerplexityAdapter(),
            "deepseek": DeepSeekAdapter(),
            "ollama": OllamaAdapter(),
        }

    def get_adapter(self, provider: str) -> object:
        """Returns adapter or raises clean exception."""
        if provider not in self.adapters:
            logger.warning("Unknown provider: {}", provider)
            raise ValueError(f"Unknown provider: {provider}")
        return self.adapters[provider]

    def get_available_providers(self) -> list[str]:
        """Returns list of available provider names."""
        return list(self.adapters.keys())
