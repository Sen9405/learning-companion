"""Tests for the eval module — 4 eval types."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from learning_companion.eval import (
    eval_end_state,
    eval_single_turn,
    eval_trajectory,
    load_golden,
)


# ---------------------------------------------------------------------------
# Load golden
# ---------------------------------------------------------------------------


class TestLoadGolden:
    def test_load_valid_json(self, tmp_path):
        data = [
            {"id": "test-1", "name": "Test", "input": {"text": "hello"}, "expected": {}}
        ]
        f = tmp_path / "golden.json"
        f.write_text(json.dumps(data, ensure_ascii=False))
        result = load_golden(str(f))
        assert len(result) == 1
        assert result[0]["id"] == "test-1"

    def test_load_empty(self, tmp_path):
        f = tmp_path / "empty.json"
        f.write_text("[]")
        assert load_golden(str(f)) == []

    def test_load_missing_file(self):
        try:
            load_golden("/nonexistent/golden.json")
            assert False, "Should have raised"
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Eval type 1: single-turn
# ---------------------------------------------------------------------------


class TestEvalSingleTurn:
    """Basic checks on a single run output."""

    def test_min_note_length_pass(self):
        state = {"note": "x" * 500, "concepts": [], "questions": []}
        expected = {"min_note_length": 200}
        r = eval_single_turn(state, expected)
        assert r["score"] == 1.0
        assert r["passed"] == 1

    def test_min_note_length_fail(self):
        state = {"note": "short", "concepts": [], "questions": []}
        expected = {"min_note_length": 500}
        r = eval_single_turn(state, expected)
        assert r["score"] == 0.0
        assert "note too short" in r["issues"][0]

    def test_must_have_concepts_pass(self):
        state = {"note": "x" * 100, "concepts": [{"name": "AI"}], "questions": []}
        expected = {"min_note_length": 0, "must_have_concepts": True}
        r = eval_single_turn(state, expected)
        assert r["passed"] == 2

    def test_must_have_concepts_fail(self):
        state = {"note": "x" * 100, "concepts": [], "questions": []}
        expected = {"min_note_length": 0, "must_have_concepts": True}
        r = eval_single_turn(state, expected)
        assert "no concepts extracted" in r["issues"]

    def test_must_have_questions_pass(self):
        state = {"note": "x" * 100, "concepts": [], "questions": [{"q": "?"}]}
        expected = {"min_note_length": 0, "must_have_questions": True}
        r = eval_single_turn(state, expected)
        assert r["passed"] == 2

    def test_must_have_questions_fail(self):
        state = {"note": "x" * 100, "concepts": [], "questions": []}
        expected = {"min_note_length": 0, "must_have_questions": True}
        r = eval_single_turn(state, expected)
        assert "no questions extracted" in r["issues"]

    def test_language_en_pass(self):
        state = {"note": "This is English text", "concepts": [], "questions": []}
        expected = {"min_note_length": 0, "language": "en"}
        r = eval_single_turn(state, expected)
        assert r["passed"] == 2

    def test_language_en_fail(self):
        state = {"note": "Это русский текст", "concepts": [], "questions": []}
        expected = {"min_note_length": 0, "language": "en"}
        r = eval_single_turn(state, expected)
        assert "note has cyrillic" in r["issues"][0]

    def test_all_passed(self):
        state = {
            "note": "x" * 1000,
            "concepts": [{"name": "AI"}],
            "questions": [{"q": "?"}],
        }
        expected = {
            "min_note_length": 500,
            "must_have_concepts": True,
            "must_have_questions": True,
        }
        r = eval_single_turn(state, expected)
        assert r["score"] == 1.0
        assert r["passed"] == r["total"]
        assert r["issues"] == []


# ---------------------------------------------------------------------------
# Eval type 2: trajectory
# ---------------------------------------------------------------------------


class TestEvalTrajectory:
    """Evaluate full agent trajectory."""

    def test_full_trajectory_pass(self):
        state = {
            "content": "A" * 500,
            "analysis": "B" * 300,
            "note": "C" * 300,
        }
        r = eval_trajectory(state, "input text")
        assert r["score"] == 1.0
        assert r["passed"] == r["total"]

    def test_no_content(self):
        state = {"content": "", "analysis": "", "note": ""}
        r = eval_trajectory(state, "input")
        assert r["score"] == 0.0
        assert len(r["issues"]) >= 2

    def test_analysis_too_short(self):
        state = {
            "content": "A" * 1000,
            "analysis": "B",
            "note": "C" * 100,
        }
        r = eval_trajectory(state, "input")
        # content exists (1), analysis too short (0), note exists (1), progression fails (0)
        assert r["passed"] == 2
        assert any("analysis too short" in i for i in r["issues"])

    def test_missing_note(self):
        state = {
            "content": "A" * 500,
            "analysis": "B" * 200,
            "note": "",
        }
        r = eval_trajectory(state, "input")
        assert r["passed"] <= 3
        assert any("note too short" in i for i in r["issues"])


# ---------------------------------------------------------------------------
# Eval type 3: LLM-as-judge
# ---------------------------------------------------------------------------


class TestEvalLlmJudge:
    """LLM-as-judge scoring."""

    @patch("learning_companion.eval.llm_call")
    def test_valid_judge_response(self, mock_llm_call):
        mock_llm_call.return_value = (
            '{"scores": {"completeness": 0.8, "structure": 0.9, "accuracy": 0.7, '
            '"practical_value": 0.6, "language": 1.0}, '
            '"total": 0.8, "weaknesses": ["could be more practical"]}',
            {"prompt_tokens": 50, "completion_tokens": 30},
        )
        from learning_companion.eval import eval_llm_judge

        r = eval_llm_judge("test note", "ru", "test input")
        assert r["score"] == 0.8
        assert r["details"]["language"] == 1.0
        assert len(r["weaknesses"]) == 1

    @patch("learning_companion.eval.llm_call")
    def test_malformed_json_with_braces(self, mock_llm_call):
        mock_llm_call.return_value = (
            'Some preamble {"scores": {"completeness": 0.5}, "total": 0.5, '
            '"weaknesses": []} trailing',
            {"prompt_tokens": 50, "completion_tokens": 30},
        )
        from learning_companion.eval import eval_llm_judge

        r = eval_llm_judge("test", "ru", "input")
        assert r["score"] == 0.5

    @patch("learning_companion.eval.llm_call")
    def test_totally_malformed(self, mock_llm_call):
        mock_llm_call.return_value = ("not json", {"prompt_tokens": 10, "completion_tokens": 5})
        from learning_companion.eval import eval_llm_judge

        r = eval_llm_judge("test", "ru", "input")
        assert r["score"] == 0.0
        assert "parse error" in r["weaknesses"]

    def test_empty_note(self):
        from learning_companion.eval import eval_llm_judge

        r = eval_llm_judge("", "ru", "input")
        assert r["score"] == 0.0
        assert "empty note" in r["weaknesses"]


# ---------------------------------------------------------------------------
# Eval type 4: end-state
# ---------------------------------------------------------------------------


class TestEvalEndState:
    """Check final graph state correctness."""

    def _make_full_state(self, **overrides):
        state = {
            "url": "",
            "text": "some input",
            "title": "Test",
            "content": "A" * 100,
            "analysis": "B" * 100,
            "note": "C" * 100,
            "concepts": [],
            "questions": [],
            "run_id": "eval-test",
            "source_type": "text",
            "stage": "done",
            "language": "ru",
        }
        state.update(overrides)
        return state

    def test_valid_end_state(self):
        state = self._make_full_state()
        r = eval_end_state(state)
        assert r["score"] == 1.0
        assert r["issues"] == []

    def test_unexpected_stage(self):
        state = self._make_full_state(stage="planner")
        r = eval_end_state(state)
        assert "unexpected final stage: planner" in r["issues"]

    def test_missing_note(self):
        state = self._make_full_state(note="")
        r = eval_end_state(state)
        assert "note field empty in final state" in r["issues"]

    def test_missing_state_fields(self):
        state = self._make_full_state()
        del state["content"]
        del state["analysis"]
        r = eval_end_state(state)
        assert any("missing state fields" in i for i in r["issues"])

    def test_no_source_type(self):
        state = self._make_full_state(source_type="")
        r = eval_end_state(state)
        assert "source_type not set" in r["issues"]

    def test_no_run_id(self):
        state = self._make_full_state()
        state.pop("run_id", None)
        r = eval_end_state(state)
        # score should be lower, no crash
        assert r["score"] < 1.0


# ---------------------------------------------------------------------------
# Full eval run
# ---------------------------------------------------------------------------


class TestEvalRun:
    """Integration tests for run_eval."""

    def test_run_eval_report_structure(self, tmp_path):
        """Verify full eval produces correct report structure."""
        from learning_companion.eval import run_eval

        golden = [
            {
                "id": "test-1",
                "name": "Test",
                "difficulty": "basic",
                "input": {"text": "test content", "language": "en"},
                "expected": {"min_note_length": 0},
            }
        ]
        golden_path = tmp_path / "golden.json"
        golden_path.write_text(json.dumps(golden, ensure_ascii=False))

        with (
            patch("learning_companion.eval.compile_agent") as mock_compile,
            patch("learning_companion.eval.llm_call") as mock_llm,
        ):
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = {
                "url": "",
                "text": "test content",
                "title": "Test",
                "content": "A" * 500,
                "analysis": "B" * 300,
                "note": "C" * 300,
                "concepts": [{"name": "test"}],
                "questions": [{"q": "?"}],
                "run_id": "eval-test-1",
                "source_type": "text",
                "stage": "done",
                "language": "en",
            }
            mock_compile.return_value = mock_agent
            mock_llm.return_value = (
                '{"scores": {}, "total": 0.9, "weaknesses": []}',
                {"prompt_tokens": 10, "completion_tokens": 5},
            )

            report = run_eval(str(golden_path), threshold=0.5)

        assert report["total"] == 1
        assert "eval_types" in report
        assert set(report["eval_types"].keys()) == {"single_turn", "trajectory", "llm_judge", "end_state"}
        assert "by_difficulty" in report
        assert "results" in report
        assert len(report["results"]) == 1
        result = report["results"][0]
        assert "single_turn" in result
        assert "trajectory" in result
        assert "llm_judge" in result
        assert "end_state" in result

    def test_run_eval_with_multiple_difficulties(self, tmp_path):
        """Report includes difficulty breakdown."""
        from learning_companion.eval import run_eval

        golden = [
            {"id": "b-1", "name": "Basic", "difficulty": "basic",
             "input": {"text": "hello", "language": "en"},
             "expected": {"min_note_length": 0}},
            {"id": "a-1", "name": "Advanced", "difficulty": "advanced",
             "input": {"text": "complex stuff", "language": "en"},
             "expected": {"min_note_length": 0}},
        ]
        golden_path = tmp_path / "multi.json"
        golden_path.write_text(json.dumps(golden, ensure_ascii=False))

        with (
            patch("learning_companion.eval.compile_agent") as mock_compile,
            patch("learning_companion.eval.llm_call") as mock_llm,
        ):
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = {
                "url": "", "text": "x", "title": "T",
                "content": "A" * 500, "analysis": "B" * 300, "note": "C" * 300,
                "concepts": [{"name": "t"}], "questions": [{"q": "?"}],
                "run_id": "eval-x", "source_type": "text",
                "stage": "done", "language": "en",
            }
            mock_compile.return_value = mock_agent
            mock_llm.return_value = ('{"scores": {}, "total": 0.8, "weaknesses": []}',
                                     {"prompt_tokens": 10, "completion_tokens": 5})
            report = run_eval(str(golden_path))

        assert "basic" in report["by_difficulty"]
        assert "advanced" in report["by_difficulty"]
        assert report["total"] == 2

    def test_eval_report_saved(self, tmp_path):
        """Verify eval saves report to eval_report.json."""
        from learning_companion.eval import run_eval

        golden = [
            {
                "id": "t-1", "name": "Test", "difficulty": "basic",
                "input": {"text": "x", "language": "en"},
                "expected": {"min_note_length": 0},
            }
        ]
        golden_path = tmp_path / "g.json"
        golden_path.write_text(json.dumps(golden, ensure_ascii=False))

        with (
            patch("learning_companion.eval.compile_agent") as mock_compile,
            patch("learning_companion.eval.llm_call") as mock_llm,
        ):
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = {
                "url": "", "text": "x", "title": "T",
                "content": "A" * 500, "analysis": "B" * 300, "note": "C" * 300,
                "concepts": [], "questions": [],
                "run_id": "eval-t-1", "source_type": "text",
                "stage": "done", "language": "en",
            }
            mock_compile.return_value = mock_agent
            mock_llm.return_value = ('{"scores": {}, "total": 0.7, "weaknesses": []}',
                                     {"prompt_tokens": 10, "completion_tokens": 5})
            with patch("learning_companion.eval.Path.open"):
                report = run_eval(str(golden_path))

        assert report["total"] == 1


class TestEvalCLI:
    """Tests for eval CLI integration."""

    def test_eval_command_in_cli(self):
        import sys
        from unittest.mock import patch

        from learning_companion.cli import main

        testargs = ["learning-companion", "eval", "--golden", "tests/golden.json"]
        with (
            patch.object(sys, "argv", testargs),
            patch("learning_companion.eval.run_eval", return_value={
                "total": 0, "all_passed": 0, "all_failed": 0,
                "overall_avg_score": 0.6,
                "threshold": 0.5, "eval_types": {}, "by_difficulty": {}, "results": [],
            }) as mock_eval,
        ):
            main()
            assert mock_eval.called

    def test_eval_fails_below_threshold(self):
        import sys
        from unittest.mock import patch

        import pytest

        from learning_companion.cli import main

        testargs = ["learning-companion", "eval", "--threshold", "0.7"]
        with (
            patch.object(sys, "argv", testargs),
            patch("learning_companion.eval.run_eval", return_value={
                "total": 0, "all_passed": 0, "all_failed": 0,
                "overall_avg_score": 0.5,
                "threshold": 0.7, "eval_types": {}, "by_difficulty": {}, "results": [],
            }),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 1

    def test_eval_passes_above_threshold(self):
        import sys
        from unittest.mock import patch

        from learning_companion.cli import main

        testargs = ["learning-companion", "eval", "--golden", "tests/golden.json"]
        with (
            patch.object(sys, "argv", testargs),
            patch("learning_companion.eval.run_eval", return_value={
                "total": 1, "all_passed": 1, "all_failed": 0,
                "overall_avg_score": 0.8,
                "threshold": 0.5, "eval_types": {}, "by_difficulty": {}, "results": [],
            }),
        ):
            main()
