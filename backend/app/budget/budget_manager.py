"""Budget management for session spending."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger


@dataclass
class BudgetWarning:
    """Budget warning information."""

    level: str  # "warning" (75%), "critical" (90%), "exhausted" (100%)
    message: str
    percentage_used: float
    remaining_usd: float


class BudgetManager:
    """Tracks and manages session budget."""

    def __init__(self, total_budget_usd: float) -> None:
        self.total_budget_usd = total_budget_usd
        self.used_budget_usd = 0.0
        self.remaining_budget_usd = total_budget_usd
        self.percentage_used = 0.0
        self.economic_mode = False
        self.last_request_cost = 0.0
        self.warning_threshold = 0.75
        self.critical_threshold = 0.90

        logger.info("Budget manager initialized: ${:.2f}", total_budget_usd)

    def update(self, cost_usd: float) -> BudgetWarning | None:
        """
        Update budget after a request.

        Updates exactly once per request:
        - used += cost
        - remaining = total - used
        - pct = used / total * 100
        - Check thresholds: 75% (warning), 90% (critical), 100% (exhausted)

        Returns: BudgetWarning if threshold crossed, else None
        """
        self.last_request_cost = cost_usd
        self.used_budget_usd = round(self.used_budget_usd + cost_usd, 6)
        self.remaining_budget_usd = round(self.total_budget_usd - self.used_budget_usd, 6)
        self.percentage_used = (
            round((self.used_budget_usd / self.total_budget_usd) * 100, 2)
            if self.total_budget_usd > 0
            else 0.0
        )

        logger.debug(
            "Budget updated: used=${:.6f}, remaining=${:.6f}, pct={}%",
            self.used_budget_usd,
            self.remaining_budget_usd,
            self.percentage_used,
        )

        warning = self._check_thresholds()
        if warning:
            logger.warning("{}", warning.message)
        return warning

    def _check_thresholds(self) -> BudgetWarning | None:
        """Check budget thresholds and return warning if crossed."""
        pct = self.percentage_used / 100.0

        if pct >= 1.0:
            return BudgetWarning(
                level="exhausted",
                message="Budget exhausted. Using local model only.",
                percentage_used=self.percentage_used,
                remaining_usd=self.remaining_budget_usd,
            )

        if pct >= self.critical_threshold:
            return BudgetWarning(
                level="critical",
                message=f"⚠️ CRITICAL: Budget at {self.percentage_used}%. Remaining: ${self.remaining_budget_usd:.2f}",
                percentage_used=self.percentage_used,
                remaining_usd=self.remaining_budget_usd,
            )

        if pct >= self.warning_threshold:
            return BudgetWarning(
                level="warning",
                message=f"You've used {self.percentage_used}% of your budget.",
                percentage_used=self.percentage_used,
                remaining_usd=self.remaining_budget_usd,
            )

        return None

    def enable_economic_mode(self) -> None:
        """Enable economic mode."""
        self.economic_mode = True
        logger.info("Economic Mode enabled")

    def disable_economic_mode(self) -> None:
        """Disable economic mode."""
        self.economic_mode = False
        logger.info("Economic Mode disabled")

    def is_budget_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        return self.remaining_budget_usd <= 0.0

    def get_summary(self) -> dict:
        """Get budget summary as dict."""
        return {
            "total_budget_usd": self.total_budget_usd,
            "used_budget_usd": self.used_budget_usd,
            "remaining_budget_usd": self.remaining_budget_usd,
            "percentage_used": self.percentage_used,
            "economic_mode": self.economic_mode,
            "last_request_cost": self.last_request_cost,
        }
