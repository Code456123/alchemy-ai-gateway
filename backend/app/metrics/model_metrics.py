"""Model latency metrics tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ModelMetric:
    """Latency metrics for a single model."""

    model_name: str
    request_count: int = 0
    total_latency_ms: float = 0.0
    average_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0

    def add_latency(self, latency_ms: float) -> None:
        """Add a latency measurement."""
        self.request_count += 1
        self.total_latency_ms += latency_ms
        self.average_latency_ms = round(self.total_latency_ms / self.request_count, 2)
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)


class ModelMetrics:
    """Tracks rolling average latency per model."""

    def __init__(self) -> None:
        self.metrics: dict[str, ModelMetric] = {}

    def update(self, model: str, latency_ms: float) -> None:
        """Update latency metrics for a model."""
        if model not in self.metrics:
            self.metrics[model] = ModelMetric(model_name=model)

        self.metrics[model].add_latency(latency_ms)
        logger.debug(
            "Model metrics: {} - avg latency {}ms (based on {} requests)",
            model,
            self.metrics[model].average_latency_ms,
            self.metrics[model].request_count,
        )

    def get_average_latency(self, model: str) -> float | None:
        """Return average latency or None if never used."""
        if model in self.metrics:
            return self.metrics[model].average_latency_ms
        return None

    def get_metric(self, model: str) -> ModelMetric | None:
        """Return full metric object for model."""
        return self.metrics.get(model)

    def get_all_metrics(self) -> dict[str, ModelMetric]:
        """Return all metrics."""
        return self.metrics.copy()
