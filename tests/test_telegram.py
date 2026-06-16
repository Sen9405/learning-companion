"""Tests for Telegram module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from learning_companion.telegram import (
    _escape_telegram_md,
    _strip_questions_section,
    send_telegram,
)


class TestEscapeMarkdown:
    """Test MarkdownV2 escaping for Telegram."""

    def test_escape_special_chars(self):
        text = "_*[]()~`>#+-=|{}.! hello"
        escaped = _escape_telegram_md(text)
        assert escaped == "\\_\\*\\[\\]\\(\\)\\~\\`\\>\\#\\+\\-\\=\\|\\{\\}\\.\\! hello"

    def test_escape_plain_text(self):
        text = "Hello, world! Normal text 123."
        escaped = _escape_telegram_md(text)
        # Only '!' and '.' should be escaped
        assert "!" not in text or "\\!" in escaped

    def test_escape_empty(self):
        assert _escape_telegram_md("") == ""

    def test_escape_no_special(self):
        text = "abc123"
        assert _escape_telegram_md(text) == text

    def test_escape_newlines_preserved(self):
        text = "line1\nline2"
        escaped = _escape_telegram_md(text)
        assert "line1" in escaped
        assert "line2" in escaped
        assert "\n" in escaped


class TestStripQuestions:
    """Test stripping questions section from notes."""

    def test_strip_russian_header(self):
        text = "Main content\n\n### Вопросы для проверки знаний\nQ1\nQ2"
        result = _strip_questions_section(text)
        assert "Main content" in result
        assert "Вопросы" not in result

    def test_strip_english_header(self):
        text = "Content\n\n## Questions for review\nQ1\nQ2"
        result = _strip_questions_section(text)
        assert "Content" in result
        assert "Questions" not in result

    def test_strip_bold_header(self):
        text = "Content\n\n**Вопросы для проверки**\nQ1"
        result = _strip_questions_section(text)
        assert "Content" in result
        assert "Вопросы" not in result

    def test_no_questions_section(self):
        text = "Just content\nNo questions here"
        result = _strip_questions_section(text)
        assert result == text

    def test_empty_text(self):
        assert _strip_questions_section("") == ""


class TestSendTelegram:
    """Test Telegram sending (mocked)."""

    @patch("httpx.post")
    def test_send_telegram_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}
        mock_post.return_value = mock_response

        result = send_telegram("Hello", token="test:token", chat_id="123")
        assert result is not None
        assert result["ok"] is True

    @patch("httpx.post")
    def test_send_telegram_with_markdown(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}
        mock_post.return_value = mock_response

        send_telegram("*bold* _italic_", token="test:token", chat_id="123")

        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs["json"]
        assert payload["parse_mode"] == "MarkdownV2"
        assert payload["text"] == "*bold* _italic_"

    @patch("httpx.post")
    def test_send_telegram_no_credentials(self, mock_post):
        result = send_telegram("Hello", token=None, chat_id=None)
        assert result is None
        mock_post.assert_not_called()
