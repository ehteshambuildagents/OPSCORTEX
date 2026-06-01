"""
nodes.py — LangGraph node functions.

Each node validates its output with Pydantic and appends to the append-only
audit log. A decorator wraps every node so any exception is caught, logged,
audited, and surfaced as `state["error"]` (routed to the `error` node) instead
of crashing the graph.
"""
from __future__ import annotations

import functools
from datetime import datetime, timezone
from typing import Callable

import config
import tools
from adapters import fetch_all
from detectors import run_detectors
from rag import retrieve
from state import (ActionPlan, ActionStep, Anomaly, ApprovalStatus, Citation,
                   OpsState, RiskLevel, RootCauseAnalysis, Severity, TriageResult)

log = config.get_logger("nodes")


# ---------------------------------------------------------------------------
# Node decorator: logging + error handling + safety-net audit entry
# ---------------------------------------------------------------------------
def node(name: str) -> Callable:
    def deco(fn: Callable[[OpsState], dict]) -> Callable[[OpsState], dict]:
        @functools.wraps(fn)
        def wrapper(state: OpsState) -> dict:
            log.info("[%s] start", name)
            try:
                update = fn(state) or {}
                log.info("[%s] ok", name)
                return update
            except Exception as exc:  # graceful degradation
                log.exception("[%s] failed: %s", name, exc)
                entry = tools.record_audit(name, "error", outputs=f"{type(exc).__name__}: {exc}")
                return {
                    "error": {"node": name, "type": type(exc).__name__, "message": str(exc)},
                    "audit_log": [entry],
                }
        return wrapper
    return deco


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _anomaly_payload(a: Anomaly) -> dict:
    return {
        "id": a.id, "metric": a.metric, "source": a.source,
        "severity": a.severity.value, "severity_score": a.severity_score,
        "deviation": a.deviation, "detectors": a.detectors, "direction": a.direction,
        "observed_value": a.observed_value, "baseline": a.baseline,
        "description": a.description,
    }


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
@node("ingest")
def ingest(state: OpsState) -> dict:
    scenario = state.get("scenario", "default")
    signals = fetch_all(scenario)
    sources = sorted({s.source for s in signals})
    entry = tools.record_audit(
        "ingest", "ok",
        inputs=f"scenario={scenario}",
        outputs=f"{len(signals)} signals from {len(sources)} sources: {', '.join(sources)}",
    )
    return {"signals": signals, "audit_log": [entry]}


@node("detect")
def detect(state: OpsState) -> dict:
    signals = state.get("signals", [])
    anomalies = run_detectors(signals)
    breakdown = _severity_breakdown(anomalies)
    entry = tools.record_audit(
        "detect", "ok",
        inputs=f"{len(signals)} signals",
        outputs=f"{len(anomalies)} anomalies; severity={breakdown}",
    )
    return {"anomalies": anomalies, "audit_log": [entry]}


@node("triage")
def triage(state: OpsState) -> dict:
    llm = tools.get_llm()
    results: list[TriageResult] = []
    audits, usage = [], []
    for a in state.get("anomalies", []):
        data, meta = llm.structured(tools.TASK_TRIAGE, {"anomaly": _anomaly_payload(a)},
                                    config.TRIAGE_MODEL, node="triage")
        tr = TriageResult(anomaly_id=a.id, severity=Severity(data["severity"]),
                          priority=data["priority"], rationale=data["rationale"],
                          model_used=meta["model_used"])
        results.append(tr)
        usage.append(meta)
        audits.append(tools.record_audit(
            "triage", "ok", inputs=f"anomaly={a.metric}",
            outputs=f"{tr.severity.value}/{tr.priority}", model_used=meta["model_used"]))
    return {"triage": results, "audit_log": audits, "token_usage": usage}


@node("rca")
def rca(state: OpsState) -> dict:
    llm = tools.get_llm()
    results: list[RootCauseAnalysis] = []
    audits, usage = [], []
    for a in state.get("anomalies", []):
        hits = retrieve(a.description)
        retrieved = [{"id": inc["id"], "title": inc["title"], "root_cause": inc["root_cause"],
                      "resolution": inc["resolution"], "similarity": round(score, 4)}
                     for inc, score in hits]
        rconf = retrieved[0]["similarity"] if retrieved else 0.0
        payload = {"anomaly": _anomaly_payload(a), "retrieved": retrieved,
                   "retrieval_confidence": rconf}
        data, meta = llm.structured(tools.TASK_RCA, payload, config.SYNTHESIS_MODEL, node="rca")
        analysis = RootCauseAnalysis(
            anomaly_id=a.id, hypothesis=data["hypothesis"], evidence=data.get("evidence", []),
            citations=[Citation(**c) for c in data.get("citations", [])],
            confidence=data["confidence"], low_confidence=data.get("low_confidence", False),
            retrieval_confidence=rconf, model_used=meta["model_used"])
        results.append(analysis)
        usage.append(meta)
        status = "low_confidence" if analysis.low_confidence else "ok"
        audits.append(tools.record_audit(
            "rca", status, inputs=f"anomaly={a.metric}; retrieval_conf={rconf:.2f}",
            outputs=f"conf={analysis.confidence:.2f}; cites={len(analysis.citations)}",
            model_used=meta["model_used"]))
    return {"rca": results, "audit_log": audits, "token_usage": usage}


@node("plan")
def plan(state: OpsState) -> dict:
    llm = tools.get_llm()
    triage_by = {t.anomaly_id: t for t in state.get("triage", [])}
    rca_by = {r.anomaly_id: r for r in state.get("rca", [])}
    plans: list[ActionPlan] = []
    audits, usage = [], []
    for a in state.get("anomalies", []):
        tr = triage_by.get(a.id)
        rc = rca_by.get(a.id)
        payload = {
            "anomaly": _anomaly_payload(a),
            "triage": {"severity": tr.severity.value, "priority": tr.priority} if tr else {},
            "rca": {"confidence": rc.confidence, "low_confidence": rc.low_confidence,
                    "hypothesis": rc.hypothesis} if rc else {},
        }
        data, meta = llm.structured(tools.TASK_PLAN, payload, config.SYNTHESIS_MODEL, node="plan")
        steps = [ActionStep(order=s["order"], action=s["action"], risk=RiskLevel(s["risk"]),
                            customer_facing=s.get("customer_facing", False),
                            destructive=s.get("destructive", False),
                            rationale=s.get("rationale", "")) for s in data["steps"]]
        ap = ActionPlan(anomaly_id=a.id, summary=data["summary"], steps=steps,
                        overall_risk=RiskLevel(data["overall_risk"]),
                        confidence=data["confidence"], model_used=meta["model_used"])
        plans.append(ap)
        usage.append(meta)
        audits.append(tools.record_audit(
            "plan", "ok", inputs=f"anomaly={a.metric}",
            outputs=f"{len(steps)} steps; risk={ap.overall_risk.value}; conf={ap.confidence:.2f}",
            model_used=meta["model_used"]))
    return {"plans": plans, "audit_log": audits, "token_usage": usage}


@node("route")
def route(state: OpsState) -> dict:
    anomalies_by = {a.id: a for a in state.get("anomalies", [])}
    rca_by = {r.anomaly_id: r for r in state.get("rca", [])}
    plans = state.get("plans", [])
    auto_ids, hitl_ids = [], []
    audits = []

    for p in plans:
        a = anomalies_by.get(p.anomaly_id)
        rc = rca_by.get(p.anomaly_id)
        reasons: list[str] = []
        if a and a.severity.value in config.HITL_SEVERITIES:
            reasons.append(f"severity {a.severity.value}")
        if p.overall_risk.value in config.HITL_RISK_LEVELS:
            reasons.append("high-risk step")
        if any(s.destructive for s in p.steps):
            reasons.append("destructive action")
        if any(s.customer_facing for s in p.steps):
            reasons.append("customer-facing action")
        if rc and rc.confidence < config.RCA_CONFIDENCE_CUTOFF:
            reasons.append("low RCA confidence")
        if rc and rc.low_confidence:
            reasons.append("RCA admitted uncertainty")
        if p.confidence < config.PLAN_CONFIDENCE_CUTOFF:
            reasons.append("low plan confidence")

        p.requires_approval = bool(reasons)
        p.approval_reasons = reasons
        (hitl_ids if p.requires_approval else auto_ids).append(p.id)
        audits.append(tools.record_audit(
            "route", "ok", inputs=f"plan={p.id}",
            outputs=("HITL: " + ", ".join(reasons)) if reasons else "AUTO"))

    return {"plans": plans, "auto_plan_ids": auto_ids, "hitl_plan_ids": hitl_ids,
            "audit_log": audits}


@node("auto_action")
def auto_action(state: OpsState) -> dict:
    auto_ids = set(state.get("auto_plan_ids", []))
    executed, audits = [], []
    for p in state.get("plans", []):
        if p.id not in auto_ids:
            continue
        action = _execute(p, approved_by="auto")
        executed.append(action)
        audits.append(tools.record_audit(
            "auto_action", "executed", inputs=f"plan={p.id}",
            outputs=f"{len(p.steps)} steps executed (risk={p.overall_risk.value})",
            model_used=p.model_used))
    if not executed:
        audits.append(tools.record_audit("auto_action", "skipped", outputs="no auto-eligible plans"))
    return {"executed_actions": executed, "audit_log": audits}


@node("hitl")
def hitl(state: OpsState) -> dict:
    provider = tools.get_approval_provider()
    plans_by = {p.id: p for p in state.get("plans", [])}
    rca_by = {r.anomaly_id: r for r in state.get("rca", [])}

    approvals, executed, audits = [], [], []
    for pid in state.get("hitl_plan_ids", []):
        p = plans_by[pid]
        rc = rca_by.get(p.anomaly_id)
        context = {"approval_reasons": p.approval_reasons,
                   "rca_hypothesis": rc.hypothesis if rc else None,
                   "rca_confidence": rc.confidence if rc else None}
        decision = provider.request_approval(p, context)
        approvals.append(decision)
        if decision.status == ApprovalStatus.APPROVED:
            executed.append(_execute(p, approved_by=decision.approver))
            audits.append(tools.record_audit(
                "hitl", "executed_after_approval", inputs=f"plan={p.id}",
                outputs=f"approved by {decision.approver}", model_used=p.model_used))
        else:
            audits.append(tools.record_audit(
                "hitl", decision.status.value, inputs=f"plan={p.id}",
                outputs=f"{decision.status.value} ({decision.reason})"))
    return {"approvals": approvals, "executed_actions": executed, "audit_log": audits}


@node("error")
def error(state: OpsState) -> dict:
    err = state.get("error") or {"message": "unknown error"}
    entry = tools.record_audit("error", "handled",
                               outputs=f"{err.get('node')}: {err.get('message')}")
    return {"audit_log": [entry],
            "summary": {"status": "error", "error": err}}


# ---------------------------------------------------------------------------
# Action execution (mock) + summary
# ---------------------------------------------------------------------------
def _execute(plan: ActionPlan, approved_by: str) -> dict:
    """Mock execution: in production this dispatches to real runbooks/automation.
    Here it just records that the plan's steps were carried out."""
    return {
        "plan_id": plan.id,
        "anomaly_id": plan.anomaly_id,
        "status": "executed",
        "approved_by": approved_by,
        "steps_executed": len(plan.steps),
        "overall_risk": plan.overall_risk.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _severity_breakdown(anomalies: list[Anomaly]) -> dict[str, int]:
    out: dict[str, int] = {}
    for a in anomalies:
        out[a.severity.value] = out.get(a.severity.value, 0) + 1
    return out


def build_summary(state: OpsState) -> dict:
    """Post-run summary built from final state (used by main/test_runner/web)."""
    if state.get("error"):
        return {"status": "error", "error": state["error"],
                "audit_entries": len(state.get("audit_log", []))}
    anomalies = state.get("anomalies", [])
    plans = state.get("plans", [])
    approvals = state.get("approvals", [])
    usage = state.get("token_usage", [])
    return {
        "status": "ok",
        "scenario": state.get("scenario"),
        "anomalies_found": len(anomalies),
        "severity_breakdown": _severity_breakdown(anomalies),
        "plans_drafted": len(plans),
        "auto_actions": len(state.get("auto_plan_ids", [])),
        "hitl_actions": len(state.get("hitl_plan_ids", [])),
        "approvals": {d.status.value: sum(1 for x in approvals if x.status == d.status)
                      for d in approvals},
        "executed_actions": len(state.get("executed_actions", [])),
        "low_confidence_rca": sum(1 for r in state.get("rca", []) if r.low_confidence),
        "estimated_tokens": sum(m.get("prompt_tokens", 0) + m.get("completion_tokens", 0)
                                for m in usage),
        "audit_entries": len(state.get("audit_log", [])),
    }
