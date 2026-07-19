from __future__ import annotations

import time

import httpx
from loguru import logger

from backend.app.config.settings import Settings, get_settings
from backend.app.constants.models import ModelID
from backend.app.gateway.groq_client import GroqGateway
from backend.app.gateway.mock import MockResponseEngine, MockResult
from backend.app.models.analysis import PromptAnalysis
from backend.app.models.request import PromptRequest
from backend.app.pricing import PricingService

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Provide clear, accurate, and concise responses."
)


class NvidiaGateway:
    """Calls NVIDIA's OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        fallback: MockResponseEngine | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._fallback = fallback or MockResponseEngine()

        self._base_url = getattr(self._settings, "nvidia_base_url", "").rstrip("/")
        if self._base_url.endswith("/v1"):
            self._endpoint = f"{self._base_url}/chat/completions"
        else:
            self._endpoint = f"{self._base_url}/v1/chat/completions"

        self._model = getattr(self._settings, "nvidia_model", "")
        self._api_key = getattr(self._settings, "nvidia_api_key", "")
        self._timeout = getattr(self._settings, "nvidia_timeout_ms", 10000) / 1000.0
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        self._pricing_service = PricingService()
        self._groq_gateway = GroqGateway(settings=self._settings, fallback=self._fallback)

        logger.info(
            "NvidiaGateway initialized endpoint={} model={} timeout={}s",
            self._endpoint,
            self._model,
            self._timeout,
        )

    def generate(
        self,
        request: PromptRequest,
        model: ModelID,
        analysis: PromptAnalysis | None = None,
        context_text: str = "",
    ) -> MockResult:
        """Call NVIDIA and return a result compatible with the pipeline."""
        user_content = request.prompt
        if context_text:
            user_content = f"{context_text}\n\n---\n\n{user_content}"

        messages = [
            {"role": "system", "content": _DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        logger.info(
            "Nvidia API call model={} prompt_len={} context_len={}",
            self._model,
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
            logger.info(data)
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", len(user_content) // 4)
            completion_tokens = usage.get("completion_tokens", len(text) // 4)
            provider_model = self._model or model.value
            cost_usd = self._pricing_service.calculate_cost(
                "nvidia", provider_model, prompt_tokens, completion_tokens
            )

            logger.info(
                "Nvidia API success model={} latency={:.0f}ms tokens={}+{} cost=${:.6f}",
                self._model,
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
                provider="nvidia",
                provider_model=self._model or model.value,
            )

        except httpx.HTTPStatusError as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.error(
                "Nvidia API HTTP error model={} status={} latency={:.0f}ms: {}",
                self._model,
                exc.response.status_code,
                latency_ms,
                exc.response.text[:200],
            )
            return self._retry_with_groq(request, model, analysis, context_text)

        except httpx.TimeoutException:
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.error(
                "Nvidia API timeout model={} after {:.0f}ms (limit={}s)",
                self._model,
                latency_ms,
                self._timeout,
            )
            return self._retry_with_groq(request, model, analysis, context_text)

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.opt(exception=True).error(
                "Nvidia API unexpected error model={} latency={:.0f}ms: {}",
                self._model,
                latency_ms,
                exc,
            )
            return self._fallback.generate(request, model, analysis, context_text)

    def _retry_with_groq(
        self,
        request: PromptRequest,
        model: ModelID,
        analysis: PromptAnalysis | None = None,
        context_text: str = "",
    ) -> MockResult:
        """Try Groq before falling back to the mock engine."""
        if self._has_groq_config():
            logger.warning(
                "Nvidia gateway failed for model={} — retrying with Groq gateway",
                self._model,
            )
            try:
                return self._groq_gateway.generate(request, model, analysis, context_text)
            except Exception as exc:
                logger.warning(
                    "Groq retry for NVIDIA failure also failed: {}",
                    exc,
                )

        return self._fallback.generate(request, model, analysis, context_text)

    def _has_groq_config(self) -> bool:
        return bool(
            getattr(self._settings, "groq_api_key", "")
            and getattr(self._settings, "groq_base_url", "")
            and getattr(self._settings, "groq_model", "")
        )

