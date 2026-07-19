"""Mozilla Otari gateway — model dispatch, fallback chains, response handling."""

from __future__ import annotations

from backend.app.gateway.groq_client import GroqGateway
from backend.app.gateway.mock import MockResponseEngine, MockResult
from backend.app.gateway.nvidia_client import NvidiaGateway
from backend.app.gateway.ollama_client import OllamaGateway
from backend.app.gateway.otari_client import OtariGateway, create_gateway

__all__ = [
    "MockResponseEngine",
    "MockResult",
    "OtariGateway",
    "OllamaGateway",
    "NvidiaGateway",
    "GroqGateway",
    "create_gateway",
]
