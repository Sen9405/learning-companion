"""Persistent SQLite ledger for LLM calls, cost, latency, and cache hits."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LlmCallRecord:
    """One LLM call entry stored in the run ledger."""

    run_id: str
    stage: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    latency: float
    cache_hit: bool
    error: str | None = None


class RunLedger:
    """SQLite-backed ledger for production cost observability."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser()
        self._ensure_schema()

    def record_llm_call(self, record: LlmCallRecord) -> None:
        """Persist one LLM call."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO llm_calls(
                    run_id, stage, model, prompt_tokens, completion_tokens,
                    cost, latency, cache_hit, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.stage,
                    record.model,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.cost,
                    record.latency,
                    1 if record.cache_hit else 0,
                    record.error,
                    time.time(),
                ),
            )
            conn.commit()

    def recent_calls(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return latest LLM calls as dictionaries."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, run_id, stage, model, prompt_tokens, completion_tokens,
                       cost, latency, cache_hit, error, created_at
                FROM llm_calls
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def summary(self, run_id: str | None = None) -> dict[str, Any]:
        """Aggregate cost/tokens/cache/error counts, optionally scoped to one run."""
        where = "WHERE run_id = ?" if run_id else ""
        params = (run_id,) if run_id else ()
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*) AS total_calls,
                    COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                    COALESCE(SUM(cost), 0) AS total_cost,
                    COALESCE(SUM(cache_hit), 0) AS cache_hits,
                    COALESCE(SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END), 0) AS errors,
                    COALESCE(AVG(latency), 0) AS avg_latency
                FROM llm_calls
                {where}
                """,
                params,
            ).fetchone()
        data = dict(row)
        data["total_cost"] = round(float(data["total_cost"]), 6)
        data["avg_latency"] = round(float(data["avg_latency"]), 3)
        return data

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    cost REAL NOT NULL,
                    latency REAL NOT NULL,
                    cache_hit INTEGER NOT NULL,
                    error TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_run_id ON llm_calls(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_stage ON llm_calls(stage)")
            conn.commit()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["cache_hit"] = bool(data["cache_hit"])
        return data
