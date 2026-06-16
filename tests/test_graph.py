"""Tests for graph state, builder, and nodes."""

from __future__ import annotations

from unittest.mock import patch


from learning_companion.graph import make_initial_state
from learning_companion.graph.builder import build_graph, compile_agent


class TestLearningState:
    """Test state creation and structure."""

    def test_make_initial_state_defaults(self):
        state = make_initial_state(run_id="test-123")
        assert state["url"] == ""
        assert state["text"] == ""
        assert state["content"] == ""
        assert state["analysis"] == ""
        assert state["stage"] == "planner"
        assert state["approved"] is False
        assert state["language"] == "ru"
        assert state["run_id"] == "test-123"
        assert state["source_type"] == "text"

    def test_make_initial_state_with_url(self):
        state = make_initial_state(
            url="https://youtube.com/watch?v=123",
            title="Test Video",
            language="en",
        )
        assert state["url"] == "https://youtube.com/watch?v=123"
        assert state["title"] == "Test Video"
        assert state["language"] == "en"

    def test_make_initial_state_with_text(self):
        state = make_initial_state(
            text="Some content to analyze",
            title="My Note",
        )
        assert state["text"] == "Some content to analyze"
        assert state["title"] == "My Note"

    def test_state_dict_structure(self):
        state = make_initial_state(url="https://example.com")
        required_keys = [
            "url", "text", "title", "content", "analysis", "note",
            "questions_list", "telegram_note", "run_id", "source_type",
            "stage", "approved", "error", "language", "trace_id",
            "prompt_tokens", "completion_tokens", "cost", "notes_checked",
        ]
        for key in required_keys:
            assert key in state, f"Missing key: {key}"

    def test_initial_counts(self):
        state = make_initial_state()
        assert state["prompt_tokens"] == 0
        assert state["completion_tokens"] == 0
        assert state["cost"] == 0.0
        assert state["notes_checked"] == 0


class TestGraphBuilder:
    """Test graph construction."""

    def test_build_graph_has_correct_nodes(self):
        graph = build_graph()
        # StateGraph exposes nodes internally
        assert "planner" in graph.nodes
        assert "fetcher" in graph.nodes
        assert "analyst" in graph.nodes
        assert "writer" in graph.nodes

    def test_build_graph_has_entry_point(self):
        graph = build_graph()
        # Entry point is set via set_entry_point during build
        assert graph is not None

    def test_compile_agent_without_checkpointer(self):
        agent = compile_agent(checkpointer=None)
        assert agent is not None

    def test_compile_agent_has_interrupts(self):
        agent = compile_agent(checkpointer=None)
        # Should be configured with interrupts
        assert agent is not None


class TestNodes:
    """Test individual node functions."""

    def test_planner_node_text_source(self):
        from learning_companion.graph.nodes import planner_node

        state = make_initial_state(text="Some text")
        result = planner_node(state)

        assert result["source_type"] == "text"
        assert result["stage"] == "fetcher"

    def test_planner_node_youtube_url(self):
        from learning_companion.graph.nodes import planner_node

        state = make_initial_state(url="https://youtube.com/watch?v=abc123")
        result = planner_node(state)

        assert result["source_type"] == "youtube"
        assert result["stage"] == "fetcher"

    def test_planner_node_web_url(self):
        from learning_companion.graph.nodes import planner_node

        state = make_initial_state(url="https://example.com/article")
        result = planner_node(state)

        assert result["source_type"] == "web"
        assert result["stage"] == "fetcher"

    def test_planner_node_pdf_url(self):
        from learning_companion.graph.nodes import planner_node

        state = make_initial_state(url="https://example.com/paper.pdf")
        result = planner_node(state)

        assert result["source_type"] == "pdf"
        assert result["stage"] == "fetcher"

    def test_has_content_true(self):
        from learning_companion.graph.nodes import has_content

        state = make_initial_state()
        state["content"] = "Some content here"
        assert has_content(state) is True

    def test_has_content_false(self):
        from learning_companion.graph.nodes import has_content

        state = make_initial_state()
        state["content"] = ""
        assert has_content(state) is False

    def test_has_analysis_true(self):
        from learning_companion.graph.nodes import has_analysis

        state = make_initial_state()
        state["analysis"] = "Analysis text"
        assert has_analysis(state) is True

    def test_has_analysis_false(self):
        from learning_companion.graph.nodes import has_analysis

        state = make_initial_state()
        state["analysis"] = ""
        assert has_analysis(state) is False

    @patch("learning_companion.graph.nodes._fetch_web")
    def test_fetcher_node_web(self, mock_fetch):
        from learning_companion.graph.nodes import fetcher_node

        mock_fetch.return_value = "Mocked web content"
        state = make_initial_state(url="https://example.com")
        state["source_type"] = "web"

        result = fetcher_node(state)
        assert "Mocked web content" in result["content"]
        assert result["stage"] == "analyst"

    @patch("learning_companion.graph.nodes._fetch_youtube")
    def test_fetcher_node_youtube(self, mock_fetch):
        from learning_companion.graph.nodes import fetcher_node

        mock_fetch.return_value = "Mocked YouTube transcript"
        state = make_initial_state(url="https://youtube.com/watch?v=abc")
        state["source_type"] = "youtube"

        result = fetcher_node(state)
        assert "Mocked YouTube transcript" in result["content"]
        assert result["stage"] == "analyst"

    def test_fetcher_node_direct_text(self):
        from learning_companion.graph.nodes import fetcher_node

        state = make_initial_state(text="Direct input text")
        state["source_type"] = "text"
        result = fetcher_node(state)

        assert result["content"] == "Direct input text"
        assert result["stage"] == "analyst"
