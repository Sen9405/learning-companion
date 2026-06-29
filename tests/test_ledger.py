"""Tests for persistent LLM cost ledger."""

from __future__ import annotations

from learning_companion.ledger import LlmCallRecord, RunLedger


def test_run_ledger_records_llm_call(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.sqlite3")

    ledger.record_llm_call(
        LlmCallRecord(
            run_id="run-1",
            stage="planner",
            model="deepseek-v4-flash",
            prompt_tokens=10,
            completion_tokens=20,
            cost=0.001,
            latency=1.25,
            cache_hit=False,
            error=None,
        )
    )

    rows = ledger.recent_calls(limit=10)

    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-1"
    assert rows[0]["stage"] == "planner"
    assert rows[0]["cost"] == 0.001
    assert rows[0]["cache_hit"] is False


def test_run_ledger_summary_by_run(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.sqlite3")

    ledger.record_llm_call(LlmCallRecord("run-1", "planner", "deepseek-v4-flash", 10, 20, 0.001, 1.0, False, None))
    ledger.record_llm_call(LlmCallRecord("run-1", "writer", "deepseek-v4-flash", 30, 40, 0.002, 2.0, True, None))
    ledger.record_llm_call(LlmCallRecord("run-2", "planner", "deepseek-v4-flash", 1, 2, 0.003, 3.0, False, "boom"))

    summary = ledger.summary(run_id="run-1")

    assert summary["total_calls"] == 2
    assert summary["total_cost"] == 0.003
    assert summary["prompt_tokens"] == 40
    assert summary["completion_tokens"] == 60
    assert summary["cache_hits"] == 1
    assert summary["errors"] == 0
