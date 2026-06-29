"""SQLite-backed prompt cache for deterministic LLM calls."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CachedResponse:
    """Cached LLM response payload."""

    text: str
    meta: dict[str, Any]
    created_at: float


def make_cache_key(
    *,
    model: str,
    system: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    response_format: dict | None,
    prompt_version: str = "v1",
) -> str:
    """Create a stable hash for all inputs that can affect an LLM response."""
    payload = {
        "model": model,
        "system": system,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": response_format,
        "prompt_version": prompt_version,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256(raw.encode("utf-8")).hexdigest()


class PromptCache:
    """Tiny SQLite cache for deterministic LLM responses."""

    def __init__(self, db_path: str | Path, ttl_days: int = 30) -> None:
        self.db_path = Path(db_path).expanduser()
        self.ttl_days = ttl_days
        self._ensure_schema()

    def get(self, key: str) -> CachedResponse | None:
        """Return cached response or None when missing/expired."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT text, meta_json, created_at FROM prompt_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()

        if row is None:
            return None

        text, meta_json, created_at = row
        if self._is_expired(float(created_at)):
            self.delete(key)
            return None

        return CachedResponse(text=text, meta=json.loads(meta_json), created_at=float(created_at))

    def set(self, key: str, *, text: str, meta: dict[str, Any]) -> None:
        """Store or replace a cached response."""
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO prompt_cache(cache_key, text, meta_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    text = excluded.text,
                    meta_json = excluded.meta_json,
                    created_at = excluded.created_at
                """,
                (key, text, json.dumps(meta, sort_keys=True), now),
            )
            conn.commit()

    def delete(self, key: str) -> None:
        """Delete one cache entry."""
        with self._connect() as conn:
            conn.execute("DELETE FROM prompt_cache WHERE cache_key = ?", (key,))
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_cache (
                    cache_key TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    meta_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def _is_expired(self, created_at: float) -> bool:
        if self.ttl_days <= 0:
            return True
        return (time.time() - created_at) >= self.ttl_days * 24 * 60 * 60
