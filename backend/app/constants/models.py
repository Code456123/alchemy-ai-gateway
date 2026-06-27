"""Model identifiers and cost constants as defined in the PRD."""

from __future__ import annotations

from enum import StrEnum


class ModelID(StrEnum):
    """Supported model identifiers for routing."""

    # Groq models (free API)
    GROQ_LLAMA2_13B = "groq/llama2-13b-chat"
    GROQ_MIXTRAL = "groq/mixtral-8x7b-32768"
    GROQ_LLAMA2_70B = "groq/llama2-70b-chat"

    # Legacy (kept for backwards compatibility)
    LOCAL_2B = "local_2b"
    GPT4O_MINI = "gpt4o_mini"
    GPT4O = "gpt4o"


# Cost per 1K tokens (USD)
MODEL_COSTS: dict[str, dict[str, float]] = {
    ModelID.GROQ_LLAMA2_13B: {"input": 0.0, "output": 0.0},
    ModelID.GROQ_MIXTRAL: {"input": 0.0, "output": 0.0},
    ModelID.GROQ_LLAMA2_70B: {"input": 0.0, "output": 0.0},
    ModelID.LOCAL_2B: {"input": 0.0, "output": 0.0},
    ModelID.GPT4O_MINI: {"input": 0.00015, "output": 0.00060},
    ModelID.GPT4O: {"input": 0.00250, "output": 0.01000},
}

# Fallback chain: primary → fallback1 → fallback2
MODEL_FALLBACK_CHAIN: dict[str, list[str]] = {
    ModelID.GROQ_LLAMA2_13B: [ModelID.GROQ_MIXTRAL],
    ModelID.GROQ_MIXTRAL: [ModelID.GROQ_LLAMA2_70B],
    ModelID.GROQ_LLAMA2_70B: [ModelID.GROQ_MIXTRAL],
    ModelID.GPT4O: [ModelID.GPT4O_MINI, ModelID.LOCAL_2B],
    ModelID.GPT4O_MINI: [ModelID.LOCAL_2B],
    ModelID.LOCAL_2B: [],
}
