"""Long-Term Memory — PostgreSQL-backed note storage with JSON cache."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    psycopg2 = None  # type: ignore


class LongTermMemory:
    """PostgreSQL-backed long-term memory for Learning Companion.

    Stores notes with concepts, question IDs, mastery tracking.
    Falls back to SQLite if PostgreSQL is unavailable.
    """

    def __init__(
        self,
        db_url: str | None = None,
        cache_dir: str | None = None,
    ):
        self.db_url = db_url or os.environ.get(
            "PG_DSN",
            "",
        )
        if not self.db_url:
            for socket_dir in ["/var/run/postgresql", "/run/postgresql"]:
                if os.path.exists(socket_dir):
                    self.db_url = f"postgresql://sen@/learning_companion?host={socket_dir}"
                    break
            if not self.db_url:
                self.db_url = "postgresql://localhost:5432/learning_companion"
        self._conn = None
        self._cache: dict[str, Any] | None = None
        self._cache_path = os.path.join(
            cache_dir or Path.home().as_posix(),
            ".learning_companion_cache.json",
        )
        self._use_sqlite = False
        self._sqlite_conn: sqlite3.Connection | None = None

    def _connect(self):
        if self._use_sqlite:
            return self._sqlite_connect()
        if psycopg2 is None:
            print("[LTM] psycopg2 not installed, switching to SQLite fallback")
            self._use_sqlite = True
            return self._sqlite_connect()
        try:
            self._conn = psycopg2.connect(self.db_url)
            self._conn.autocommit = True
            self._ensure_table()
        except Exception as e:
            print(f"[LTM] PostgreSQL connection failed: {e}, switching to SQLite")
            self._use_sqlite = True
            return self._sqlite_connect()
        return self._conn

    def _sqlite_connect(self):
        if self._sqlite_conn is None:
            db_path = os.path.expanduser("~/.learning_companion_ltm.db")
            self._sqlite_conn = sqlite3.connect(db_path)
            self._sqlite_conn.row_factory = sqlite3.Row
            self._ensure_sqlite_table()
        return self._sqlite_conn

    def _ensure_table(self):
        if self._conn and not self._use_sqlite:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS notes (
                        id SERIAL PRIMARY KEY,
                        title TEXT NOT NULL,
                        url TEXT,
                        summary TEXT,
                        concepts JSONB DEFAULT '[]',
                        questions JSONB DEFAULT '[]',
                        glossary JSONB DEFAULT '[]',
                        questions_asked INTEGER DEFAULT 0,
                        mastery INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT NOW(),
                        accessed_at TIMESTAMP DEFAULT NOW()
                    )
                    """
                )

    def _ensure_sqlite_table(self):
        if self._sqlite_conn:
            self._sqlite_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    url TEXT,
                    summary TEXT,
                    concepts TEXT DEFAULT '[]',
                    questions TEXT DEFAULT '[]',
                    glossary TEXT DEFAULT '[]',
                    questions_asked INTEGER DEFAULT 0,
                    mastery INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    accessed_at TEXT DEFAULT (datetime('now'))
                )
                """
            )
            self._sqlite_conn.commit()

    def add_note(
        self,
        title: str,
        url: str = "",
        summary: str = "",
        concepts: list[str] | None = None,
        questions: list[dict] | None = None,
        glossary: list[dict] | None = None,
    ) -> int:
        conn = self._connect()
        if not self._use_sqlite and psycopg2 and self._conn:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO notes (title, url, summary, concepts, questions, glossary)
                    VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
                    RETURNING id
                    """,
                    (
                        title,
                        url,
                        summary,
                        json.dumps(concepts or []),
                        json.dumps(questions or []),
                        json.dumps(glossary or []),
                    ),
                )
                note_id = cur.fetchone()[0]
        else:
            cur = conn.execute(
                """
                INSERT INTO notes (title, url, summary, concepts, questions, glossary)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    title,
                    url,
                    summary,
                    json.dumps(concepts or []),
                    json.dumps(questions or []),
                    json.dumps(glossary or []),
                ),
            )
            note_id = cur.lastrowid
            conn.commit()

        self._invalidate_cache()
        return note_id

    def get_note(self, note_id: int) -> dict[str, Any] | None:
        conn = self._connect()
        if not self._use_sqlite and psycopg2 and self._conn:
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("UPDATE notes SET accessed_at = NOW() WHERE id = %s RETURNING *", (note_id,))
                row = cur.fetchone()
                return self._row_to_dict(row) if row else None
        else:
            cur = conn.execute(
                "UPDATE notes SET accessed_at = datetime('now') WHERE id = ? RETURNING *",
                (note_id,),
            )
            row = cur.fetchone()
            conn.commit()
            return self._row_to_dict(row) if row else None

    def search_notes(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        conn = self._connect()
        like = f"%{query}%"
        if not self._use_sqlite and psycopg2 and self._conn:
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM notes
                    WHERE title ILIKE %s OR summary ILIKE %s OR concepts::text ILIKE %s
                    ORDER BY accessed_at DESC LIMIT %s
                    """,
                    (like, like, like, limit),
                )
                return [self._row_to_dict(r) for r in cur.fetchall()]
        else:
            cur = conn.execute(
                """
                SELECT * FROM notes
                WHERE title LIKE ? OR summary LIKE ? OR concepts LIKE ?
                ORDER BY accessed_at DESC LIMIT ?
                """,
                (like, like, like, limit),
            )
            return [self._row_to_dict(r) for r in cur.fetchall()]

    def get_all_concepts(self) -> list[str]:
        conn = self._connect()
        concepts: list[str] = []
        if not self._use_sqlite and psycopg2 and self._conn:
            with self._conn.cursor() as cur:
                cur.execute("SELECT DISTINCT jsonb_array_elements_text(concepts) AS c FROM notes")
                concepts = [r[0] for r in cur.fetchall() if r[0]]
        else:
            rows = conn.execute("SELECT concepts FROM notes WHERE concepts != '[]'").fetchall()
            for row in rows:
                try:
                    concepts.extend(json.loads(row[0]))
                except (json.JSONDecodeError, IndexError):
                    pass
        return sorted(set(c.strip() for c in concepts if c and c.strip()))

    def get_recent_notes(self, limit: int = 10) -> list[dict[str, Any]]:
        conn = self._connect()
        if not self._use_sqlite and psycopg2 and self._conn:
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM notes ORDER BY accessed_at DESC LIMIT %s", (limit,))
                return [self._row_to_dict(r) for r in cur.fetchall()]
        else:
            cur = conn.execute("SELECT * FROM notes ORDER BY accessed_at DESC LIMIT ?", (limit,))
            return [self._row_to_dict(r) for r in cur.fetchall()]

    def update_mastery(self, note_id: int, increment: int = 1) -> None:
        conn = self._connect()
        if not self._use_sqlite and psycopg2 and self._conn:
            with self._conn.cursor() as cur:
                cur.execute(
                    "UPDATE notes SET mastery = GREATEST(0, mastery + %s) WHERE id = %s",
                    (increment, note_id),
                )
        else:
            conn.execute(
                "UPDATE notes SET mastery = MAX(0, mastery + ?) WHERE id = ?",
                (increment, note_id),
            )
            conn.commit()

    def mark_questions_asked(self, note_id: int) -> None:
        conn = self._connect()
        if not self._use_sqlite and psycopg2 and self._conn:
            with self._conn.cursor() as cur:
                cur.execute(
                    "UPDATE notes SET questions_asked = questions_asked + 1 WHERE id = %s",
                    (note_id,),
                )
        else:
            conn.execute(
                "UPDATE notes SET questions_asked = questions_asked + 1 WHERE id = ?",
                (note_id,),
            )
            conn.commit()

    def _row_to_dict(self, row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            d = dict(row)
        else:
            keys = [
                "id", "title", "url", "summary", "concepts",
                "questions", "glossary", "questions_asked", "mastery",
                "created_at", "accessed_at",
            ]
            d = dict(zip(keys, row))
        # Parse JSON fields
        for field in ("concepts", "questions", "glossary"):
            if isinstance(d.get(field), str):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    d[field] = []
        return d

    def _invalidate_cache(self) -> None:
        self._cache = None

    def _refresh_cache(self) -> None:
        """Build in-memory cache from recent notes."""
        self._cache = {
            "recent_notes": self.get_recent_notes(10),
            "known_concepts": self.get_all_concepts(),
            "note_count": len(self._cache.get("recent_notes", []))
                if self._cache else 0,
        }

    def load_cache(self) -> dict[str, Any]:
        """Return cached context — rebuilds from DB only once per session."""
        if self._cache is None:
            self._refresh_cache()
        return self._cache or {}

    def format_context(self) -> str:
        """Build a formatted string of LTM context for the planner/system prompt."""
        cache = self.load_cache()
        concepts = cache.get("known_concepts", [])
        recent = cache.get("recent_notes", [])
        lines: list[str] = []
        if concepts:
            lines.append("## Known Concepts (from LTM)")
            for c in concepts[:20]:
                lines.append(f"- {c}")
        if recent:
            lines.append(f"\n## Recent Notes ({len(recent)})")
            for n in recent:
                lines.append(f"- [{n.get('id', '?')}] {n.get('title', 'Untitled')} "
                            f"(mastery: {n.get('mastery', 0)})")
        return "\n".join(lines) if lines else "No LTM context yet."


# Singleton
_ltm_instance: LongTermMemory | None = None


def get_ltm(db_url: str | None = None) -> LongTermMemory:
    global _ltm_instance
    if _ltm_instance is None:
        _ltm_instance = LongTermMemory(db_url=db_url)
    return _ltm_instance
