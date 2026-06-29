"""Operational alerts derived from the LLM run ledger."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from learning_companion.settings import Settings


class AlertSeverity(StrEnum):
    """Alert severities for run reports."""

    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class LedgerAlert:
    """One alert emitted from ledger summary data."""

    code: str
    severity: AlertSeverity
    message: str
    value: float | int
    threshold: float | int

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
        }


def evaluate_ledger_alerts(summary: dict[str, Any], settings: Settings) -> list[LedgerAlert]:
    """Evaluate ledger summary against configured alert thresholds."""
    alerts: list[LedgerAlert] = []
    total_calls = int(summary.get("total_calls", 0) or 0)
    total_cost = float(summary.get("total_cost", 0) or 0)
    errors = int(summary.get("errors", 0) or 0)
    avg_latency = float(summary.get("avg_latency", 0) or 0)
    cache_hits = int(summary.get("cache_hits", 0) or 0)
    cache_rate = (cache_hits / total_calls) if total_calls else 1.0

    if total_cost > settings.alert_cost_usd:
        alerts.append(LedgerAlert(
            "cost_threshold_exceeded",
            AlertSeverity.CRITICAL,
            f"LLM cost ${total_cost:.6f} exceeded alert threshold ${settings.alert_cost_usd:.6f}",
            round(total_cost, 6),
            settings.alert_cost_usd,
        ))
    if errors > settings.alert_error_count:
        alerts.append(LedgerAlert(
            "llm_errors_detected",
            AlertSeverity.CRITICAL,
            f"LLM errors {errors} exceeded allowed count {settings.alert_error_count}",
            errors,
            settings.alert_error_count,
        ))
    if avg_latency > settings.alert_avg_latency_seconds:
        alerts.append(LedgerAlert(
            "latency_threshold_exceeded",
            AlertSeverity.WARNING,
            f"Average LLM latency {avg_latency:.3f}s exceeded {settings.alert_avg_latency_seconds:.3f}s",
            round(avg_latency, 3),
            settings.alert_avg_latency_seconds,
        ))
    if total_calls and cache_rate < settings.alert_min_cache_hit_rate:
        alerts.append(LedgerAlert(
            "cache_hit_rate_low",
            AlertSeverity.WARNING,
            f"Cache hit rate {cache_rate:.1%} below {settings.alert_min_cache_hit_rate:.1%}",
            round(cache_rate, 3),
            settings.alert_min_cache_hit_rate,
        ))
    return alerts
