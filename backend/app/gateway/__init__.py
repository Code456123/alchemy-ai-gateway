"""Mozilla Otari gateway — model dispatch, fallback chains, response handling."""

from __future__ import annotations

from backend.app.gateway.mock import MockResponseEngine, MockResult
from backend.app.gateway.otari_client import OtariGateway, create_gateway

__all__ = ["MockResponseEngine", "MockResult", "OtariGateway", "create_gateway"]
