"""
LangGraph pipeline — wires the four agents into a directed state machine.

Flow:
  observation_node
       ↓
  research_node      (blocked if observation errored)
       ↓
  decision_node      (blocked if research errored)
       ↓
  implementation_node

Each node returns a partial state dict; LangGraph merges it with the
existing state automatically (reducer = last-write-wins on each key).
"""
from __future__ import annotations
import uuid
from typing import Any

from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.observation_agent import observation_node
from agents.research_agent import research_node
from agents.decision_agent import decision_node
from agents.implementation_agent import implementation_node


# ── Edge conditions ───────────────────────────────────────────────────────────

def _should_continue(state: dict[str, Any]) -> str:
    """Route to END on any unrecovered error."""
    return END if state.get("error") else "continue"


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph() -> Any:
    graph = StateGraph(dict)     # use plain dict; AgentState validates on entry

    graph.add_node("observation",      observation_node)
    graph.add_node("research",         research_node)
    graph.add_node("decision",         decision_node)
    graph.add_node("implementation",   implementation_node)

    # Set entry point
    graph.set_entry_point("observation")

    # Error-aware edges
    graph.add_conditional_edges(
        "observation",
        _should_continue,
        {"continue": "research", END: END},
    )
    graph.add_conditional_edges(
        "research",
        _should_continue,
        {"continue": "decision", END: END},
    )
    graph.add_conditional_edges(
        "decision",
        _should_continue,
        {"continue": "implementation", END: END},
    )
    graph.add_edge("implementation", END)

    return graph.compile()


# Module-level compiled graph (import this from the API layer)
pipeline = build_graph()


# ── Convenience runner ────────────────────────────────────────────────────────

def run_pipeline(symbol: str, run_id: str | None = None) -> AgentState:
    """
    Run the full four-agent pipeline for a given symbol.
    Returns the final AgentState.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())

    initial_state = AgentState(run_id=run_id, symbol=symbol).model_dump()
    final_state   = pipeline.invoke(initial_state)
    return AgentState(**final_state)
