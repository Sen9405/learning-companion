"""Tests for CLI argument parsing and entry points."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from learning_companion.cli import main, run_report
from learning_companion.ledger import LlmCallRecord, RunLedger
from learning_companion.settings import reset_settings_cache


class TestCLIArgs:
    """Test CLI argument parsing with argparse."""

    def test_run_requires_url_or_text(self):
        """run command without --url or --text should exit."""
        testargs = ["learning-companion", "run"]
        with patch.object(sys, "argv", testargs), pytest.raises(SystemExit):
            main()

    def test_run_with_url(self):
        testargs = ["learning-companion", "run", "--url", "https://example.com"]
        with (
            patch.object(sys, "argv", testargs),
            patch("learning_companion.cli.run_agent", return_value={
                "run_id": "test-123", "cost": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0},
            }) as mock_run,
        ):
            main()
            mock_run.assert_called_once()
            kwargs = mock_run.call_args[1]
            assert kwargs["url"] == "https://example.com"
            assert kwargs["text"] == ""

    def test_run_with_text(self):
        testargs = ["learning-companion", "run", "--text", "Some text content"]
        with (
            patch.object(sys, "argv", testargs),
            patch("learning_companion.cli.run_agent", return_value={
                "run_id": "test-123", "cost": {"prompt_tokens": 5, "completion_tokens": 10, "cost": 0.001},
            }) as mock_run,
        ):
            main()
            mock_run.assert_called_once()
            kwargs = mock_run.call_args[1]
            assert kwargs["text"] == "Some text content"
            assert kwargs["url"] == ""

    def test_run_with_title(self):
        testargs = ["learning-companion", "run", "--text", "x", "--title", "My Note"]
        with (
            patch.object(sys, "argv", testargs),
            patch("learning_companion.cli.run_agent") as mock_run,
        ):
            mock_run.return_value = {"run_id": "t", "cost": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0}}
            main()
            mock_run.assert_called_once()
            kwargs = mock_run.call_args[1]
            assert kwargs["title"] == "My Note"

    def test_run_with_language(self):
        testargs = ["learning-companion", "run", "--text", "x", "--language", "en"]
        with (
            patch.object(sys, "argv", testargs),
            patch("learning_companion.cli.run_agent") as mock_run,
        ):
            mock_run.return_value = {"run_id": "t", "cost": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0}}
            main()
            kwargs = mock_run.call_args[1]
            assert kwargs["language"] == "en"

    def test_run_with_tracing(self):
        testargs = ["learning-companion", "run", "--text", "x", "--trace"]
        with (
            patch.object(sys, "argv", testargs),
            patch("learning_companion.cli.run_agent") as mock_run,
        ):
            mock_run.return_value = {"run_id": "t", "cost": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0}}
            main()
            kwargs = mock_run.call_args[1]
            assert kwargs["enable_tracing"] is True

    def test_run_with_no_hitl(self):
        testargs = ["learning-companion", "run", "--text", "x", "--no-hitl"]
        with (
            patch.object(sys, "argv", testargs),
            patch("learning_companion.cli.run_agent") as mock_run,
        ):
            mock_run.return_value = {"run_id": "t", "cost": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0}}
            main()
            kwargs = mock_run.call_args[1]
            assert kwargs["skip_hitl"] is True

    def test_resume_requires_run_id(self):
        testargs = ["learning-companion", "resume"]
        with patch.object(sys, "argv", testargs), pytest.raises(SystemExit):
            main()

    def test_resume_with_run_id(self):
        testargs = ["learning-companion", "resume", "abc123"]
        with (
            patch.object(sys, "argv", testargs),
            patch("learning_companion.cli.resume_agent") as mock_resume,
        ):
            mock_resume.return_value = {"run_id": "abc123", "cost": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0}}
            main()
            mock_resume.assert_called_once_with("abc123", "resume-analyst")

    def test_resume_with_action(self):
        testargs = ["learning-companion", "resume", "abc123", "save-writer"]
        with (
            patch.object(sys, "argv", testargs),
            patch("learning_companion.cli.resume_agent") as mock_resume,
        ):
            mock_resume.return_value = {"run_id": "abc123", "cost": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0}}
            main()
            mock_resume.assert_called_once_with("abc123", "save-writer")

    def test_check_with_limit(self):
        testargs = ["learning-companion", "check", "--limit", "10"]
        with (
            patch.object(sys, "argv", testargs),
            patch("learning_companion.cli.run_check") as mock_check,
        ):
            main()
            mock_check.assert_called_once_with(limit=10)

    def test_check_default_limit(self):
        testargs = ["learning-companion", "check"]
        with (
            patch.object(sys, "argv", testargs),
            patch("learning_companion.cli.run_check") as mock_check,
        ):
            main()
            mock_check.assert_called_once_with(limit=5)

    def test_no_command_shows_help(self):
        testargs = ["learning-companion"]
        with patch.object(sys, "argv", testargs), patch(
            "argparse.ArgumentParser.print_help"
        ) as mock_help:
            main()
            mock_help.assert_called_once()

    def test_help_flag(self):
        testargs = ["learning-companion", "--help"]
        with patch.object(sys, "argv", testargs), pytest.raises(SystemExit):
            main()


def test_run_report_prints_ledger_summary(tmp_path, monkeypatch, capsys):
    ledger_db = tmp_path / "ledger.sqlite3"
    monkeypatch.setenv("LC_RUN_LEDGER_DB", str(ledger_db))
    reset_settings_cache()

    ledger = RunLedger(ledger_db)
    ledger.record_llm_call(LlmCallRecord("run-1", "planner", "deepseek-v4-flash", 10, 20, 0.001, 1.0, False, None))

    run_report(limit=5)

    output = capsys.readouterr().out
    assert "LLM Ledger Report" in output
    assert "Total calls: 1" in output
    assert "planner" in output
