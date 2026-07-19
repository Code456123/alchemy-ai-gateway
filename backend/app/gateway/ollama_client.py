from __future__ import annotations

import time

import httpx
from loguru import logger

from backend.app.config.settings import Settings, get_settings
from backend.app.constants.models import ModelID
from backend.app.gateway.mock import MockResponseEngine, MockResult
from backend.app.models.analysis import PromptAnalysis
from backend.app.models.request import PromptRequest
from backend.app.pricing import PricingService

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Provide clear, accurate, and concise responses."
)


class OllamaGateway:
    """Calls a local Ollama server through the /api/generate endpoint."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        fallback: MockResponseEngine | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._fallback = fallback or MockResponseEngine()

        self._base_url = self._settings.ollama_base_url.rstrip("/")
        self._endpoint = f"{self._base_url}/api/generate"
        self._model = self._settings.ollama_model
        self._timeout = self._settings.ollama_timeout_ms / 1000.0
        self._pricing_service = PricingService()

        logger.info(
            "OllamaGateway initialized model={} endpoint={} timeout={}s",
            self._model,
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
        """Call the local Ollama server and return a pipeline-compatible result."""
        user_prompt = request.prompt

        if context_text:
            user_prompt = f"""
Relevant Context:

{context_text}

User Question:
{request.prompt}
"""

        prompt = f"""
System:
{_DEFAULT_SYSTEM_PROMPT}

Instructions:
- Use the relevant context if it helps.
- If the context is not sufficient, answer using your own knowledge.
- Never say "The context does not contain..." unless the user specifically asks about the context.

{user_prompt}

Assistant:
"""

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
            },
        }

        logger.info(
            "Ollama API call model={} prompt_len={} context_len={}",
            self._model,
            len(request.prompt),
            len(context_text),
        )

        start = time.perf_counter()

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(self._endpoint, json=payload)
                response.raise_for_status()
                data = response.json()

            latency_ms = (time.perf_counter() - start) * 1000.0

            text = self._parse_response(data)

            prompt_tokens = self._estimate_tokens(prompt)
            completion_tokens = self._estimate_tokens(text)
            provider_model = self._model or model.value
            cost_usd = self._pricing_service.calculate_cost(
                "ollama", provider_model, prompt_tokens, completion_tokens
            )

            return MockResult(
                text=text,
                model=model,
                latency_ms=round(latency_ms, 3),
                cost_usd=cost_usd,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                provider="ollama",
                provider_model=self._model or model.value,
            )

        except Exception as exc:
            logger.exception(exc)
            return self._fallback.generate(request, model, analysis)

    @staticmethod
    def _parse_response(data: dict) -> str:
        if isinstance(data.get("response"), str):
            return data["response"]

        outputs = data.get("outputs")
        if isinstance(outputs, list) and outputs:
            first_output = outputs[0]
            content = first_output.get("content")
            if isinstance(content, list) and content:
                first_content = content[0]
                if isinstance(first_content, dict) and "text" in first_content:
                    return str(first_content["text"])
                if isinstance(first_content, str):
                    return first_content

        raise ValueError("Unexpected Ollama /api/generate response format")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

