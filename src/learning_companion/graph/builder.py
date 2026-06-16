"""Graph builder — compile and configure the LangGraph agent."""

from __future__ import annotations

from typing import Any

try:
    from langgraph.checkpoint.postgres import PostgresSaver
    HAS_POSTGRES = True
except ImportError:
    PostgresSaver = None
    HAS_POSTGRES = False
from langgraph.constants import END
from langgraph.graph import StateGraph

from learning_companion.graph import LearningState
from learning_companion.graph.nodes import (
    analyst_node,
    fetcher_node,
    has_content,
    planner_node,
    writer_node,
)


def build_graph() -> StateGraph:
    """Build the Learning Companion LangGraph."""
    builder = StateGraph(LearningState)

    # Add nodes
    builder.add_node("planner", planner_node)
    builder.add_node("fetcher", fetcher_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("writer", writer_node)

    # Add edges
    builder.set_entry_point("planner")
    builder.add_edge("planner", "fetcher")
    builder.add_conditional_edges("fetcher", has_content, {True: "analyst", False: END})
    builder.add_edge("analyst", "writer")
    builder.add_edge("writer", END)

    return builder


def compile_agent(
    checkpointer: Any | None = None,
) -> Any:
    """Compile the agent with optional PostgresSaver persistence."""
    graph = build_graph()
    return graph.compile(checkpointer=checkpointer, interrupt_before=["analyst", "writer"])
