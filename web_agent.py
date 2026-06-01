"""
web_agent.py — Conversational OpsCortex agent for the browser chat UI.

A real Claude model (SYNTHESIS_MODEL) drives the conversation and is given
tools that call into the OpsCortex LangGraph pipeline:

  - list_scenarios            : what monitored situations can be run
  - run_ops_pipeline          : run ingest->detect->triage->rca->plan->route
  - list_pending_approvals    : high-risk plans queued for human approval (HITL)
  - decide_approval           : approve/deny a queued plan (this IS the HITL UI)

Responses stream token-by-token. Tool execution happens between streamed
turns. Each browser session keeps its own conversation + ops context in memory.

This uses the real Anthropic API and therefore carries normal LLM risks
(latency, cost, occasional malformed output). Tool errors are caught and
returned to the model so it can explain them rather than crashing the stream.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

import anthropic

import config
import tools
from adapters import SCENARIO_PROFILES
from graph import build_graph, run
from nodes import _execute, build_summary

log = config.get_logger("web_agent")

_CLIENT = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
_APP = build_graph()  # compiled once; reused across runs

SCENARIO_DESCRIPTIONS = {
    "default": "Default demo (payment failure spike).",
    "payment_failure_spike": "Payment failure-rate spike (high severity, expect HITL).",
    "ticket_surge": "Support ticket surge (medium severity, expect auto action plan).",
    "latency_creep": "Slow API latency creep (low severity, expect auto monitor).",
    "noisy_benign": "Noisy-but-benign fluctuation (should raise NO anomaly).",
    "low_retrieval_confidence": "Novel failure mode with weak historical match "
                                "(agent must admit uncertainty).",
    "all_normal": "All sources nominal (no anomalies).",
}

SYSTEM = (
    "You are OpsCortex Assistant, an operations-intelligence agent for SaaS / "
    "e-commerce / fintech ops teams. You help engineers investigate operational "
    "anomalies across payments, support, app metrics, and infrastructure.\n\n"
    "You have tools that run a real pipeline: deterministic anomaly detection, "
    "RAG-grounded root-cause analysis over historical incidents, confidence-scored "
    "action planning, and human-in-the-loop (HITL) approval routing.\n\n"
    "Rules:\n"
    "- Ground every root-cause claim in the retrieved incidents and cite them by "
    "ID. If retrieval confidence is low, say so plainly — never invent a cause.\n"
    "- Report confidence scores honestly. Make no claims of certainty or "
    "'zero hallucination'.\n"
    "- NEVER approve destructive or customer-facing actions yourself. When a plan "
    "requires approval, explain the risk and ask the human to approve or deny via "
    "the approval controls. Only call decide_approval when the user explicitly "
    "approves or denies a specific plan.\n"
    "- Be concise and action-oriented. Use short sections and bullet points.\n"
    "- When the user asks to 'check', 'monitor', or 'investigate' something, pick "
    "the most relevant scenario and run the pipeline."
)

TOOLS: list[dict] = [
    {
        "name": "list_scenarios",
        "description": "List the operational scenarios that can be monitored/run.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_ops_pipeline",
        "description": "Run the OpsCortex pipeline for a scenario. Returns detected "
                       "anomalies, RAG-grounded root causes with citations and "
                       "confidence, action plans with risk, and routing (auto vs "
                       "HITL). Plans needing approval are queued, not executed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "scenario": {
                    "type": "string",
                    "enum": list(SCENARIO_DESCRIPTIONS.keys()),
                    "description": "Which scenario to run.",
                }
            },
            "required": ["scenario"],
        },
    },
    {
        "name": "list_pending_approvals",
        "description": "List action plans from the latest run that are queued for "
                       "human approval (HITL) and not yet decided.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "decide_approval",
        "description": "Approve or deny a specific queued plan. Approving executes "
                       "the plan (mock execution) and records it to the audit trail. "
                       "Only call this when the human explicitly decides.",
        "input_schema": {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string"},
                "approve": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["plan_id", "approve"],
        },
    },
]


@dataclass
class OpsSession:
    session_id: str
    messages: list[dict] = field(default_factory=list)
    plans_by_id: dict[str, Any] = field(default_factory=dict)
    anomalies_by_plan: dict[str, Any] = field(default_factory=dict)
    rca_by_anomaly: dict[str, Any] = field(default_factory=dict)
    decisions: dict[str, str] = field(default_factory=dict)
    run_count: int = 0


_SESSIONS: dict[str, OpsSession] = {}


def get_session(session_id: str | None) -> OpsSession:
    sid = session_id or uuid.uuid4().hex
    if sid not in _SESSIONS:
        _SESSIONS[sid] = OpsSession(session_id=sid)
    return _SESSIONS[sid]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def _tool_list_scenarios() -> dict:
    return {"scenarios": [{"name": n, "description": d}
                          for n, d in SCENARIO_DESCRIPTIONS.items()]}


def _tool_run_pipeline(session: OpsSession, scenario: str) -> dict:
    if scenario not in SCENARIO_PROFILES:
        return {"error": f"unknown scenario '{scenario}'",
                "known": list(SCENARIO_PROFILES.keys())}
    # Queue HITL plans so a human decides via the chat (do not auto-approve).
    tools.set_approval_provider(tools.QueueApprovalProvider())
    session.run_count += 1
    thread_id = f"{session.session_id}-{scenario}-{session.run_count}"
    state = run(scenario, thread_id=thread_id, app=_APP)

    if state.get("error"):
        return {"scenario": scenario, "status": "error", "error": state["error"]}

    rca_by = {r.anomaly_id: r for r in state.get("rca", [])}
    session.rca_by_anomaly.update(rca_by)

    anomalies = [{
        "id": a.id, "metric": a.metric, "source": a.source,
        "severity": a.severity.value, "observed_value": a.observed_value,
        "baseline": a.baseline, "detectors": a.detectors,
        "description": a.description,
    } for a in state.get("anomalies", [])]

    root_causes = [{
        "anomaly_id": r.anomaly_id, "hypothesis": r.hypothesis,
        "confidence": round(r.confidence, 2), "low_confidence": r.low_confidence,
        "retrieval_confidence": round(r.retrieval_confidence, 2),
        "citations": [{"id": c.incident_id, "title": c.title,
                       "similarity": round(c.similarity, 2)} for c in r.citations],
    } for r in state.get("rca", [])]

    plans = []
    for p in state.get("plans", []):
        session.plans_by_id[p.id] = p
        plans.append({
            "id": p.id, "anomaly_id": p.anomaly_id, "summary": p.summary,
            "overall_risk": p.overall_risk.value, "confidence": round(p.confidence, 2),
            "requires_approval": p.requires_approval,
            "approval_reasons": p.approval_reasons,
            "steps": [{"order": s.order, "action": s.action, "risk": s.risk.value,
                       "customer_facing": s.customer_facing, "destructive": s.destructive}
                      for s in p.steps],
        })

    return {
        "scenario": scenario,
        "status": "ok",
        "anomalies": anomalies,
        "root_causes": root_causes,
        "plans": plans,
        "routing": {"auto": state.get("auto_plan_ids", []),
                    "hitl": state.get("hitl_plan_ids", [])},
        "summary": build_summary(state),
    }


def _tool_list_pending(session: OpsSession) -> dict:
    pending = []
    for pid, p in session.plans_by_id.items():
        if p.requires_approval and pid not in session.decisions:
            pending.append({
                "id": pid, "summary": p.summary,
                "overall_risk": p.overall_risk.value,
                "approval_reasons": p.approval_reasons,
                "steps": [{"order": s.order, "action": s.action, "risk": s.risk.value,
                           "customer_facing": s.customer_facing,
                           "destructive": s.destructive} for s in p.steps],
            })
    return {"pending_approvals": pending, "count": len(pending)}


def _tool_decide(session: OpsSession, plan_id: str, approve: bool,
                 reason: str = "") -> dict:
    p = session.plans_by_id.get(plan_id)
    if p is None:
        return {"error": f"unknown plan_id '{plan_id}'",
                "known": list(session.plans_by_id.keys())}
    if plan_id in session.decisions:
        return {"error": f"plan {plan_id} already {session.decisions[plan_id]}"}
    if approve:
        action = _execute(p, approved_by="web-user")
        tools.record_audit("hitl", "executed_after_approval",
                           inputs=f"plan={plan_id}",
                           outputs=f"approved via web UI: {reason or 'n/a'}",
                           model_used=p.model_used)
        session.decisions[plan_id] = "approved"
        return {"status": "approved_and_executed", "plan_id": plan_id, "action": action}
    tools.record_audit("hitl", "denied", inputs=f"plan={plan_id}",
                       outputs=f"denied via web UI: {reason or 'n/a'}")
    session.decisions[plan_id] = "denied"
    return {"status": "denied", "plan_id": plan_id, "reason": reason}


def _execute_tool(session: OpsSession, name: str, tool_input: dict) -> dict:
    try:
        if name == "list_scenarios":
            return _tool_list_scenarios()
        if name == "run_ops_pipeline":
            return _tool_run_pipeline(session, tool_input.get("scenario", "default"))
        if name == "list_pending_approvals":
            return _tool_list_pending(session)
        if name == "decide_approval":
            return _tool_decide(session, tool_input["plan_id"],
                                bool(tool_input.get("approve")),
                                tool_input.get("reason", ""))
        return {"error": f"unknown tool '{name}'"}
    except Exception as exc:  # never crash the stream on a tool failure
        log.exception("tool %s failed", name)
        return {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Streaming agent loop
# ---------------------------------------------------------------------------
def _sse(event_type: str, payload: dict) -> str:
    return "data: " + json.dumps({"type": event_type, **payload}) + "\n\n"


def stream_agent(session: OpsSession, user_text: str) -> Iterator[str]:
    """Run one agent turn (possibly multiple tool round-trips) and yield SSE
    strings: status / text / tool / done / error events."""
    session.messages.append({"role": "user", "content": user_text})
    safety_limit = 6  # cap tool round-trips per user message
    try:
        for _ in range(safety_limit):
            with _CLIENT.messages.stream(
                model=config.SYNTHESIS_MODEL,
                max_tokens=1600,
                system=SYSTEM,
                tools=TOOLS,
                messages=session.messages,
            ) as stream:
                for delta in stream.text_stream:
                    yield _sse("text", {"delta": delta})
                final = stream.get_final_message()

            session.messages.append({"role": "assistant", "content": final.content})

            if final.stop_reason != "tool_use":
                yield _sse("done", {})
                return

            tool_results = []
            for block in final.content:
                if block.type == "tool_use":
                    yield _sse("status", {"text": f"running `{block.name}`…"})
                    result = _execute_tool(session, block.name, block.input or {})
                    yield _sse("tool", {"name": block.name, "result": result})
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                    })
            session.messages.append({"role": "user", "content": tool_results})

        yield _sse("text", {"delta": "\n\n_(stopped after reaching the tool-call "
                                     "limit for this message.)_"})
        yield _sse("done", {})
    except Exception as exc:
        log.exception("agent stream failed")
        yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})
