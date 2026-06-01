"""
graph.py — Assemble and compile the OpsCortex LangGraph.

Flow:
    ingest -> detect -> (anomalies?) -> triage -> rca -> plan -> route
                  |no                                               |
                  v                                                 v
                 END                                      auto_action -> (hitl?) -> END
    any node error -> error -> END

Uses a MemorySaver checkpointer; every run is keyed by a thread_id.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

import nodes
from state import OpsState


# ---------------------------------------------------------------------------
# Conditional routers
# ---------------------------------------------------------------------------
def _after_ingest(s: OpsState) -> str:
    return "error" if s.get("error") else "detect"


def _after_detect(s: OpsState) -> str:
    if s.get("error"):
        return "error"
    return "triage" if s.get("anomalies") else "end"


def _after_triage(s: OpsState) -> str:
    return "error" if s.get("error") else "rca"


def _after_rca(s: OpsState) -> str:
    return "error" if s.get("error") else "plan"


def _after_plan(s: OpsState) -> str:
    return "error" if s.get("error") else "route"


def _after_route(s: OpsState) -> str:
    return "error" if s.get("error") else "auto_action"


def _after_auto(s: OpsState) -> str:
    if s.get("error"):
        return "error"
    return "hitl" if s.get("hitl_plan_ids") else "end"


def _after_hitl(s: OpsState) -> str:
    return "error" if s.get("error") else "end"


# ---------------------------------------------------------------------------
# Build / compile
# ---------------------------------------------------------------------------
def build_graph(checkpointer: MemorySaver | None = None):
    """Construct and compile the graph. Returns a compiled LangGraph app."""
    g = StateGraph(OpsState)

    g.add_node("ingest", nodes.ingest)
    g.add_node("detect", nodes.detect)
    g.add_node("triage", nodes.triage)
    g.add_node("rca", nodes.rca)
    g.add_node("plan", nodes.plan)
    g.add_node("route", nodes.route)
    g.add_node("auto_action", nodes.auto_action)
    g.add_node("hitl", nodes.hitl)
    g.add_node("error", nodes.error)

    g.add_edge(START, "ingest")
    g.add_conditional_edges("ingest", _after_ingest, {"detect": "detect", "error": "error"})
    g.add_conditional_edges("detect", _after_detect,
                            {"triage": "triage", "end": END, "error": "error"})
    g.add_conditional_edges("triage", _after_triage, {"rca": "rca", "error": "error"})
    g.add_conditional_edges("rca", _after_rca, {"plan": "plan", "error": "error"})
    g.add_conditional_edges("plan", _after_plan, {"route": "route", "error": "error"})
    g.add_conditional_edges("route", _after_route,
                            {"auto_action": "auto_action", "error": "error"})
    g.add_conditional_edges("auto_action", _after_auto,
                            {"hitl": "hitl", "end": END, "error": "error"})
    g.add_conditional_edges("hitl", _after_hitl, {"end": END, "error": "error"})
    g.add_edge("error", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())


def initial_state(scenario: str, thread_id: str) -> OpsState:
    """Seed state. The additive channels must start as empty lists."""
    return {
        "scenario": scenario,
        "thread_id": thread_id,
        "audit_log": [],
        "token_usage": [],
        "approvals": [],
        "executed_actions": [],
    }


def run(scenario: str = "default", thread_id: str | None = None, app=None) -> OpsState:
    """Run one scenario end-to-end and return the final state."""
    app = app or build_graph()
    thread_id = thread_id or f"run-{scenario}"
    cfg = {"configurable": {"thread_id": thread_id}}
    return app.invoke(initial_state(scenario, thread_id), config=cfg)


if __name__ == "__main__":
    app = build_graph()
    print("graph compiled OK:", app is not None)
