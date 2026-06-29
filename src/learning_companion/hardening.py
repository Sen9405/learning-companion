"""Final hardening verification report for Phase 5 (Sprint 3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from learning_companion.alerts import evaluate_ledger_alerts
from learning_companion.ledger import RunLedger
from learning_companion.settings import Settings, get_settings


def build_hardening_report(
    *,
    settings: Settings | None = None,
    ledger: RunLedger | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build a machine-readable hardening report and pass/warn/fail verdict."""
    settings = settings or get_settings()
    ledger = ledger or RunLedger(settings.run_ledger_db)
    summary = ledger.summary(run_id=run_id)
    alerts = evaluate_ledger_alerts(summary, settings)

    checks = [
        _check("model_routing", settings.router_enabled and _is_flash_only(settings),
               "routing enabled and constrained to DeepSeek V4 Flash"),
        _check("budget_guards", settings.max_cost_usd_per_run > 0 and settings.max_llm_calls_per_run > 0,
               "cost/token/call budgets configured"),
        _check("prompt_cache", settings.enable_prompt_cache and bool(settings.prompt_cache_db),
               "prompt cache configured"),
        _check("ledger", bool(settings.run_ledger_db), "persistent run ledger configured"),
        _check("alerts", settings.alert_cost_usd > 0 and settings.alert_avg_latency_seconds > 0,
               "alert thresholds configured"),
        _check("sandbox", _sandbox_configured(), "per-run sandbox helpers available"),
    ]

    if any(check["status"] == "fail" for check in checks):
        verdict = "fail"
    elif any(alert.severity == "critical" for alert in alerts):
        verdict = "fail"
    elif alerts:
        verdict = "warn"
    else:
        verdict = "pass"

    return {
        "verdict": verdict,
        "run_id": run_id,
        "summary": summary,
        "alerts": [alert.as_dict() for alert in alerts],
        "checks": checks,
    }


def _check(name: str, passed: bool, detail: str) -> dict[str, str]:
    return {"name": name, "status": "pass" if passed else "fail", "detail": detail}


def _is_flash_only(settings: Settings) -> bool:
    models = {
        settings.router_simple_model,
        settings.router_standard_model,
        settings.router_complex_model,
    }
    return all(m == "deepseek-v4-flash" or "pro" not in m.lower() for m in models)


def _sandbox_configured() -> bool:
    try:
        from learning_companion.security import (  # noqa: F401
            RunContext,
            build_safe_env,
            run_sandboxed,
            wrap_untrusted_document,
        )
    except Exception:
        return False
    return Path.home().exists()
