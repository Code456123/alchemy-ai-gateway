"""Mozilla Otari Gateway — real LLM calls via OpenAI-compatible API.

Sends prompts to Mozilla Otari's hosted models through its OpenAI-compatible
chat completions endpoint. Falls back to the mock engine on API errors so the
pipeline never crashes due to a transient upstream failure.
"""

from __future__ import annotations

import time

import httpx
from loguru import logger

from backend.app.config.settings import Settings, get_settings
from backend.app.constants.models import ModelID
from backend.app.gateway.groq_client import GroqGateway
from backend.app.gateway.mock import MockResponseEngine, MockResult
from backend.app.gateway.nvidia_client import NvidiaGateway
from backend.app.gateway.ollama_client import OllamaGateway
from backend.app.models.analysis import PromptAnalysis
from backend.app.models.request import PromptRequest
from backend.app.pricing import PricingService

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Provide clear, accurate, and concise responses."
)


class OtariGateway:
    """Calls Mozilla Otari's OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        fallback: MockResponseEngine | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._fallback = fallback or MockResponseEngine()

        base = self._settings.otari_base_url.rstrip("/")
        if base.endswith("/v1"):
            self._endpoint = f"{base}/chat/completions"
        else:
            self._endpoint = f"{base}/v1/chat/completions"
        self._timeout = self._settings.otari_timeout_ms / 1000.0
        self._headers = {
            "Authorization": f"Bearer {self._settings.otari_api_key}",
            "Content-Type": "application/json",
        }
        self._pricing_service = PricingService()

        logger.info(
            "OtariGateway initialized endpoint={} timeout={}s",
            self._endpoint,
            self._timeout,
        )

    def generate(
        self,
        request: PromptRequest,
        model: ModelID,
        analysis: PromptAnalysis | None = None,
        context_text: str = "",
    ) -> MockResult:
        """Call Mozilla Otari and return a result compatible with the pipeline.

        Args:
            request: The inbound prompt request.
            model: The model selected by the routing engine.
            analysis: Optional task analysis (unused by the API, logged).
            context_text: Conversation context from unified memory to prepend.

        Returns:
            A :class:`MockResult` populated from the real API response.
            Falls back to the mock engine on any failure.
        """
        user_content = request.prompt
        if context_text:
            user_content = f"{context_text}\n\n---\n\n{user_content}"

        messages = [
            {"role": "system", "content": _DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        payload = {
            "model": model.value,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        logger.info(
            "Otari API call model={} prompt_len={} context_len={}",
            model.value,
            len(request.prompt),
            len(context_text),
        )

        start = time.perf_counter()
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    self._endpoint,
                    headers=self._headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            latency_ms = (time.perf_counter() - start) * 1000.0

            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", len(user_content) // 4)
            completion_tokens = usage.get("completion_tokens", len(text) // 4)
            cost_usd = self._pricing_service.calculate_cost(
                "otari", model.value, prompt_tokens, completion_tokens
            )

            logger.info(
                "Otari API success model={} latency={:.0f}ms tokens={}+{} cost=${:.6f}",
                model.value,
                latency_ms,
                prompt_tokens,
                completion_tokens,
                cost_usd,
            )

            return MockResult(
                text=text,
                model=model,
                latency_ms=round(latency_ms, 3),
                cost_usd=cost_usd,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                provider="otari",
                provider_model=model.value,
            )

        except httpx.HTTPStatusError as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.error(
                "Otari API HTTP error model={} status={} latency={:.0f}ms: {}",
                model.value,
                exc.response.status_code,
                latency_ms,
                exc.response.text[:200],
            )
            return self._fallback.generate(request, model, analysis)

        except httpx.TimeoutException:
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.error(
                "Otari API timeout model={} after {:.0f}ms (limit={}s)",
                model.value,
                latency_ms,
                self._timeout,
            )
            return self._fallback.generate(request, model, analysis)

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.opt(exception=True).error(
                "Otari API unexpected error model={} latency={:.0f}ms: {}",
                model.value,
                latency_ms,
                exc,
            )
            return self._fallback.generate(request, model, analysis)



def _is_ollama_reachable(base_url: str, timeout_seconds: float) -> bool:
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.get(f"{base_url.rstrip('/')}/api/status")
            response.raise_for_status()
        return True
    except Exception:
        return False


class ProviderAwareGateway:
    """Gateway wrapper that routes on selected model to Ollama or NVIDIA."""

    def __init__(
        self,
        ollama: OllamaGateway | None = None,
        nvidia: NvidiaGateway | None = None,
        fallback: MockResponseEngine | None = None,
    ) -> None:
        self._ollama = ollama
        self._nvidia = nvidia
        self._fallback = fallback or MockResponseEngine()

    def generate(
        self,
        request: PromptRequest,
        model: ModelID,
        analysis: PromptAnalysis | None = None,
        context_text: str = "",
    ) -> MockResult:
        if model is ModelID.LOCAL_2B and self._ollama is not None:
            return self._ollama.generate(request, model, analysis, context_text)

        if model in {ModelID.GPT4O_MINI, ModelID.GPT4O} and self._nvidia is not None:
            return self._nvidia.generate(request, model, analysis, context_text)

        if self._ollama is not None:
            return self._ollama.generate(request, model, analysis, context_text)
        if self._nvidia is not None:
            return self._nvidia.generate(request, model, analysis, context_text)
        return self._fallback.generate(request, model, analysis, context_text)


def create_gateway(
    settings: Settings | None = None,
) -> OllamaGateway | OtariGateway | NvidiaGateway | ProviderAwareGateway | MockResponseEngine:
    """Factory: choose the first available gateway implementation.

    Selection order:
        1. Ollama local server if reachable
        2. NVIDIA cloud when configured
        3. Otari gateway when configured
        4. Mock response engine otherwise
    """
    resolved = settings or get_settings()

    has_nvidia = bool(resolved.nvidia_api_key and resolved.nvidia_base_url)
    has_ollama = bool(resolved.ollama_base_url and _is_ollama_reachable(
        resolved.ollama_base_url,
        min(resolved.ollama_timeout_ms / 1000.0, 3.0),
    ))

    if has_ollama and has_nvidia:
        logger.info("Gateway: ProviderAwareGateway (Ollama + NVIDIA available)")
        return ProviderAwareGateway(
            ollama=OllamaGateway(settings=resolved),
            nvidia=NvidiaGateway(settings=resolved),
        )

    if has_ollama:
        logger.info("Gateway: OllamaGateway (local Ollama reachable)")
        return OllamaGateway(settings=resolved)

    if has_nvidia:
        logger.info("Gateway: NvidiaGateway (live Nvidia API)")
        return NvidiaGateway(settings=resolved)

    if resolved.groq_api_key and resolved.groq_base_url and resolved.groq_model:
        logger.info("Gateway: GroqGateway (live API)")
        return GroqGateway(settings=resolved)

    if resolved.otari_api_key and resolved.otari_base_url:
        logger.info("Gateway: OtariGateway (live API)")
        return OtariGateway(settings=resolved)

    logger.info("Gateway: MockResponseEngine (no Ollama/Otari available)")
    return MockResponseEngine()
