"""Tests for Long-Term Memory module."""

from __future__ import annotations

import tempfile

import pytest

from learning_companion.memory import LongTermMemory


@pytest.fixture
def ltm():
    """Create a LongTermMemory with SQLite :memory: fallback and temp cache dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ltm = LongTermMemory(
            db_url="sqlite://",  # will fail and switch to SQLite
            cache_dir=tmpdir,
        )
        # Force SQLite mode and create a fresh in-memory DB
        ltm._use_sqlite = True
        import sqlite3
        ltm._sqlite_conn = sqlite3.connect(":memory:")
        ltm._sqlite_conn.row_factory = sqlite3.Row
        ltm._ensure_sqlite_table()
        yield ltm


class TestLongTermMemory:
    """Test LTM using SQLite fallback (no PostgreSQL needed)."""

    def test_add_and_get_note(self, ltm):
        note_id = ltm.add_note(
            title="Test Note",
            url="https://example.com",
            summary="A test summary",
            concepts=["python", "testing"],
            questions=[{"q": "What?", "a": "42"}],
            glossary=[{"term": "test", "definition": "a check"}],
        )
        assert note_id > 0

        note = ltm.get_note(note_id)
        assert note is not None
        assert note["title"] == "Test Note"
        assert note["url"] == "https://example.com"
        assert "python" in note["concepts"]
        assert len(note["questions"]) == 1
        assert note["questions"][0]["q"] == "What?"

    def test_get_nonexistent_note(self, ltm):
        note = ltm.get_note(99999)
        assert note is None

    def test_search_notes(self, ltm):
        ltm.add_note(title="Python Basics", summary="Learning Python")
        ltm.add_note(title="Java Basics", summary="Learning Java")

        results = ltm.search_notes("Python")
        assert len(results) >= 1
        assert results[0]["title"] == "Python Basics"

        results = ltm.search_notes("Java")
        assert len(results) >= 1

    def test_search_notes_no_match(self, ltm):
        ltm.add_note(title="Python")
        results = ltm.search_notes("Rust")
        assert len(results) == 0

    def test_get_all_concepts(self, ltm):
        ltm.add_note(title="A", concepts=["python", "testing"])
        ltm.add_note(title="B", concepts=["python", "langgraph"])

        concepts = ltm.get_all_concepts()
        assert "python" in concepts
        assert "testing" in concepts
        assert "langgraph" in concepts
        assert len(concepts) == 3

    def test_get_recent_notes(self, ltm):
        for i in range(5):
            ltm.add_note(title=f"Note {i}")

        recent = ltm.get_recent_notes(limit=3)
        assert len(recent) <= 3

        recent_all = ltm.get_recent_notes(limit=10)
        assert len(recent_all) >= 5

    def test_update_mastery(self, ltm):
        note_id = ltm.add_note(title="Mastery Test")
        ltm.update_mastery(note_id, increment=1)
        note = ltm.get_note(note_id)
        assert note["mastery"] == 1

        ltm.update_mastery(note_id, increment=-1)
        note = ltm.get_note(note_id)
        assert note["mastery"] == 0

    def test_mark_questions_asked(self, ltm):
        note_id = ltm.add_note(title="Questions Test")
        ltm.mark_questions_asked(note_id)
        note = ltm.get_note(note_id)
        assert note["questions_asked"] == 1

        ltm.mark_questions_asked(note_id)
        note = ltm.get_note(note_id)
        assert note["questions_asked"] == 2

    def test_format_context_empty(self, ltm):
        context = ltm.format_context()
        assert context == "No LTM context yet."

    def test_format_context_with_notes(self, ltm):
        ltm.add_note(title="Context Test", concepts=["test-concept"])
        context = ltm.format_context()
        assert "Context Test" in context
        assert "test-concept" in context
        assert "Known Concepts" in context
        assert "Recent Notes" in context

    def test_load_cache(self, ltm):
        ltm.add_note(title="Cache Test", concepts=["caching"])
        cache = ltm.load_cache()
        assert "known_concepts" in cache
        assert "recent_notes" in cache
        assert "caching" in cache["known_concepts"]

    def test_json_field_parsing(self, ltm):
        """Test that JSON fields are properly parsed."""
        note_id = ltm.add_note(
            title="JSON Test",
            concepts=["a", "b"],
            questions=[{"q": "Q1", "a": "A1"}],
        )

        note = ltm.get_note(note_id)
        assert isinstance(note["concepts"], list)
        assert isinstance(note["questions"], list)
        assert isinstance(note["glossary"], list)

    def test_multiple_notes_with_same_concept(self, ltm):
        ltm.add_note(title="N1", concepts=["python"])
        ltm.add_note(title="N2", concepts=["python"])
        ltm.add_note(title="N3", concepts=["java"])

        concepts = ltm.get_all_concepts()
        assert concepts == ["java", "python"]  # sorted, unique
