"""Tests for the LLM client module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from learning_companion.llm import get_cost, llm_call, reset_counters


class TestCostTracking:
    """Test token and cost tracking counters."""

    def test_reset_counters(self):
        reset_counters()
        cost = get_cost()
        assert cost["prompt_tokens"] == 0
        assert cost["completion_tokens"] == 0
        assert cost["cost"] == 0.0

    def test_get_cost_returns_dict(self):
        reset_counters()
        cost = get_cost()
        assert isinstance(cost, dict)
        assert "prompt_tokens" in cost
        assert "completion_tokens" in cost
        assert "total_tokens" in cost
        assert "cost" in cost

    def test_cost_structure(self):
        reset_counters()
        cost = get_cost()
        assert cost["total_tokens"] == cost["prompt_tokens"] + cost["completion_tokens"]


class TestLlMCall:
    """Test LLM calling with mocked API."""

    @patch("learning_companion.llm.OpenAI")
    def test_llm_call_returns_text_and_meta(self, mock_openai):
        reset_counters()
        import learning_companion.llm as llm_mod
        llm_mod._client = None
        llm_mod.DEEPSEEK_API_KEY = "test-key"

        # Mock the API response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello, world!"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_client.chat.completions.create.return_value = mock_response

        text, meta = llm_call("system prompt", [{"role": "user", "content": "hi"}])

        assert text == "Hello, world!"
        assert meta["prompt_tokens"] == 10
        assert meta["completion_tokens"] == 20
        assert meta["cost"] > 0
        assert meta["latency"] >= 0

        # Check global cost tracking
        cost = get_cost()
        assert cost["prompt_tokens"] == 10
        assert cost["completion_tokens"] == 20

    @patch("learning_companion.llm.OpenAI")
    def test_llm_call_empty_response(self, mock_openai):
        reset_counters()
        from learning_companion.llm import _client as llm_client
        if llm_client is not None:
            import learning_companion.llm as llm_mod
            llm_mod._client = None

        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = None

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 0

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_client.chat.completions.create.return_value = mock_response

        import learning_companion.llm as llm_mod
        llm_mod.DEEPSEEK_API_KEY = "test-key"
        text, meta = llm_call("test", [])
        assert text == ""

    @patch("learning_companion.llm.OpenAI")
    def test_llm_call_with_temperature(self, mock_openai):
        reset_counters()
        import learning_companion.llm as llm_mod
        llm_mod._client = None
        llm_mod.DEEPSEEK_API_KEY = "test-key"

        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 5

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_response.usage = mock_usage

        mock_client.chat.completions.create.return_value = mock_response

        import learning_companion.llm as llm_mod
        llm_mod.DEEPSEEK_API_KEY = "test-key"

        llm_call("sys", [], temperature=0.7)

        # Verify temperature was passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7
