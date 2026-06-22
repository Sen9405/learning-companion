"""Tests for Inspect AI benchmark integration."""

from __future__ import annotations

import json
from unittest.mock import patch


class TestInspectDataset:
    """Test golden → Inspect Sample conversion."""

    def test_golden_to_samples_simple(self, tmp_path):
        from learning_companion.inspect_benchmark import _golden_to_samples

        data = [
            {
                "id": "t-1",
                "name": "Test",
                "difficulty": "basic",
                "input": {"text": "hello world", "title": "Test", "language": "en"},
                "expected": {"min_note_length": 100},
            }
        ]
        f = tmp_path / "g.json"
        f.write_text(json.dumps(data, ensure_ascii=False))

        samples = _golden_to_samples(str(f))
        assert len(samples) == 1
        s = samples[0]
        assert s.id == "t-1"
        assert s.input == "hello world"
        assert s.target == "en"
        assert s.metadata["title"] == "Test"
        assert s.metadata["difficulty"] == "basic"

    def test_golden_to_samples_multiple(self, tmp_path):
        from learning_companion.inspect_benchmark import _golden_to_samples

        data = [
            {"id": "a", "name": "A", "difficulty": "basic",
             "input": {"text": "t1", "title": "", "language": "ru"},
             "expected": {}},
            {"id": "b", "name": "B", "difficulty": "advanced",
             "input": {"text": "t2", "title": "", "language": "en"},
             "expected": {}},
        ]
        f = tmp_path / "g.json"
        f.write_text(json.dumps(data, ensure_ascii=False))

        samples = _golden_to_samples(str(f))
        assert len(samples) == 2
        assert samples[0].id == "a"
        assert samples[1].id == "b"

    def test_golden_empty_dataset(self, tmp_path):
        from learning_companion.inspect_benchmark import _golden_to_samples

        f = tmp_path / "empty.json"
        f.write_text("[]")
        assert _golden_to_samples(str(f)) == []


class TestInspectCLI:
    """Test CLI integration for inspect-benchmark."""

    def test_inspect_benchmark_command_in_cli(self):
        import sys

        from learning_companion.cli import main

        testargs = ["learning-companion", "inspect-benchmark", "--golden", "tests/golden.json"]
        with (
            patch.object(sys, "argv", testargs),
            patch("learning_companion.inspect_benchmark.run_inspect_benchmark") as mock_bench,
        ):
            main()
            assert mock_bench.called

    def test_inspect_benchmark_missing_dep(self):
        """Should show helpful error if inspect-ai not installed."""
        import sys

        from learning_companion.cli import main

        testargs = ["learning-companion", "inspect-benchmark"]
        with (
            patch.object(sys, "argv", testargs),
            patch("learning_companion.inspect_benchmark.run_inspect_benchmark", side_effect=ImportError("no module named inspect_ai")),
        ):
            try:
                main()
                assert False, "Should have exited"
            except SystemExit as e:
                assert e.code == 1
