"""Sprint 3 tests for model routing, alerts, and final hardening report."""

from __future__ import annotations

from learning_companion.alerts import AlertSeverity, evaluate_ledger_alerts
from learning_companion.hardening import build_hardening_report
from learning_companion.ledger import LlmCallRecord, RunLedger
from learning_companion.routing import ModelRouter
from learning_companion.settings import Settings


def test_model_router_classifies_prompt_complexity_without_using_pro_model():
    settings = Settings.from_env({
        "LC_ROUTER_SIMPLE_MAX_CHARS": "1000",
        "LC_ROUTER_COMPLEX_MIN_CHARS": "4000",
        "LC_ROUTER_COMPLEX_MODEL": "deepseek-v4-pro",  # user may misconfigure, router must stay cost-safe by default
    })
    router = ModelRouter(settings)

    simple = router.route(stage="planner", messages=[{"role": "user", "content": "short"}], max_tokens=4096, temperature=0.3)
    complex_route = router.route(
        stage="analyst.merge",
        messages=[{"role": "user", "content": "x" * 5000}],
        max_tokens=8192,
        temperature=0.7,
    )

    assert simple.complexity == "simple"
    assert simple.model == "deepseek-v4-flash"
    assert simple.cache is True
    assert complex_route.complexity == "complex"
    assert complex_route.model == "deepseek-v4-flash"
    assert complex_route.max_tokens <= settings.router_complex_max_tokens
    assert complex_route.temperature <= 0.3


def test_ledger_alerts_flag_cost_errors_latency_and_cache_rate(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    ledger.record_llm_call(LlmCallRecord("run-1", "planner", "deepseek-v4-flash", 1000, 100, 0.020, 10.0, False, None))
    ledger.record_llm_call(LlmCallRecord("run-1", "writer", "deepseek-v4-flash", 1000, 100, 0.020, 12.0, False, "boom"))

    settings = Settings.from_env({
        "LC_ALERT_COST_USD": "0.01",
        "LC_ALERT_ERROR_COUNT": "0",
        "LC_ALERT_AVG_LATENCY_SECONDS": "5",
        "LC_ALERT_MIN_CACHE_HIT_RATE": "0.5",
    })

    alerts = evaluate_ledger_alerts(ledger.summary("run-1"), settings)
    codes = {alert.code for alert in alerts}

    assert "cost_threshold_exceeded" in codes
    assert "llm_errors_detected" in codes
    assert "latency_threshold_exceeded" in codes
    assert "cache_hit_rate_low" in codes
    assert all(alert.severity in {AlertSeverity.WARNING, AlertSeverity.CRITICAL} for alert in alerts)


def test_hardening_report_includes_routing_alerts_security_and_verdict(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    ledger.record_llm_call(LlmCallRecord("run-1", "planner", "deepseek-v4-flash", 10, 5, 0.0001, 0.2, True, None))
    settings = Settings.from_env({
        "LC_RUN_LEDGER_DB": str(tmp_path / "ledger.sqlite3"),
        "LC_ENABLE_PROMPT_CACHE": "true",
        "LC_ALERT_COST_USD": "0.05",
    })

    report = build_hardening_report(settings=settings, ledger=ledger, run_id="run-1")

    assert report["verdict"] == "pass"
    check_names = {check["name"] for check in report["checks"]}
    assert {"model_routing", "budget_guards", "prompt_cache", "ledger", "alerts", "sandbox"}.issubset(check_names)
    assert report["summary"]["total_calls"] == 1
    assert report["alerts"] == []
