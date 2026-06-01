"""
tools.py — Cross-cutting utilities: the LLM client (offline + Anthropic),
robust JSON parsing, token budgeting, the append-only audit helper, and the
non-TTY human-approval (HITL) interface.

The LLM is accessed only through `LLMClient.structured(...)`. Both backends
return the same dict shape, which nodes validate with Pydantic. The offline
backend is deterministic and derives its output from the detector facts in the
payload — it validates wiring/governance, not model-quality reasoning.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import tenacity

import config
from state import ActionPlan, ApprovalDecision, ApprovalStatus, AuditEntry

log = config.get_logger("tools")

# Task identifiers used to dispatch both backends.
TASK_TRIAGE = "triage"
TASK_RCA = "rca"
TASK_PLAN = "plan"


# ===========================================================================
# Robust JSON parsing
# ===========================================================================
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip())


def _extract_outermost_object(text: str) -> Optional[str]:
    """Return the outermost balanced {...} block, or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def parse_json(text: str) -> dict:
    """Parse model output into a dict. Strip markdown fences, then fall back to
    regex-extracting the outermost {...} block. Surface the real error on total
    failure — never silently return junk."""
    cleaned = _strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as first_err:
        block = _extract_outermost_object(cleaned)
        if block is not None:
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                pass
        snippet = text[:300].replace("\n", "\\n")
        raise ValueError(
            f"Could not parse JSON from model output ({first_err}). "
            f"Raw output starts with: {snippet!r}"
        ) from first_err


# ===========================================================================
# Token budgeting (crude char/4 estimate)
# ===========================================================================
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // config.CHARS_PER_TOKEN)


def enforce_budget(node: str, payload: dict) -> tuple[dict, bool]:
    """Trim oversized payloads to the per-node token budget. Returns the
    (possibly trimmed) payload and a `truncated` flag. We trim the retrieved
    incident list first, since it is the largest and most compressible field."""
    budget = config.TOKEN_BUDGETS.get(node)
    if not budget:
        return payload, False
    if estimate_tokens(json.dumps(payload, default=str)) <= budget:
        return payload, False

    trimmed = dict(payload)
    truncated = False
    retrieved = trimmed.get("retrieved")
    if isinstance(retrieved, list):
        while retrieved and estimate_tokens(json.dumps(trimmed, default=str)) > budget:
            retrieved = retrieved[:-1]
            trimmed["retrieved"] = retrieved
            truncated = True
    return trimmed, truncated


# ===========================================================================
# Offline deterministic backend
# ===========================================================================
_PRIORITY_BY_SEVERITY = {"CRITICAL": "P0", "HIGH": "P1", "MEDIUM": "P2", "LOW": "P3"}

# Per-metric action-step templates: (action, risk, customer_facing, destructive)
_PLAN_TEMPLATES: dict[str, list[tuple[str, str, bool, bool]]] = {
    "payment_failure_rate": [
        ("Page the on-call payments engineer and open a Sev incident", "medium", False, False),
        ("Check payment-processor status and recent fraud-rule/credential changes", "low", False, False),
        ("Fail over to the secondary payment processor", "high", False, True),
        ("Pause automatic charge retries to limit customer impact", "high", True, True),
    ],
    "ticket_volume": [
        ("Cluster incoming tickets by root cause", "low", False, False),
        ("Notify the support lead and staff up the queue", "low", False, False),
        ("Prepare an internal knowledge-base note for agents", "low", False, False),
    ],
    "p95_latency_ms": [
        ("Increase monitoring frequency and add a latency dashboard", "low", False, False),
        ("Review recent deploys and slow database queries", "low", False, False),
    ],
    "error_rate": [
        ("Notify the on-call engineer and inspect recent deploys", "medium", False, False),
        ("Roll back the most recent deploy if error rate correlates with it", "high", False, True),
    ],
    "uptime_pct": [
        ("Page the on-call SRE and check availability zones", "medium", False, False),
        ("Shift traffic away from unhealthy nodes/zones", "high", False, True),
    ],
    "churn_rate": [
        ("Open a retention review with product and customer-success teams", "low", False, False),
    ],
    "payment_volume": [
        ("Verify the checkout flow across browsers and recent front-end deploys", "medium", False, False),
    ],
    "reco_ctr": [
        ("Notify the ML / recommendations team", "low", False, False),
        ("Compare the live model version and feature pipeline against last-known-good", "low", False, False),
    ],
}
_DEFAULT_TEMPLATE = [("Investigate the anomaly and notify the owning team", "low", False, False)]
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def _offline_triage(payload: dict) -> dict:
    a = payload["anomaly"]
    sev = a["severity"]
    return {
        "severity": sev,
        "priority": _PRIORITY_BY_SEVERITY.get(sev, "P3"),
        "rationale": (
            f"{a['metric']} on {a['source']} shows a {a['direction']} deviation "
            f"(severity score {a['severity_score']:.2f}) flagged by "
            f"{', '.join(a['detectors'])}; classified {sev}."
        ),
    }


def _offline_rca(payload: dict) -> dict:
    a = payload["anomaly"]
    retrieved = payload.get("retrieved", [])
    rconf = float(payload.get("retrieval_confidence", 0.0))
    citations = [{"incident_id": r["id"], "title": r["title"],
                  "similarity": round(float(r["similarity"]), 4)} for r in retrieved]
    detector_evidence = [
        f"{a['metric']} observed {a['observed_value']:g} vs baseline ~{a['baseline']:g}",
        f"deterministic detectors fired: {', '.join(a['detectors'])}",
    ]

    if rconf < config.RAG_LOW_RETRIEVAL_CUTOFF or not retrieved:
        # Admit uncertainty rather than invent a cause.
        return {
            "hypothesis": (
                "Insufficient historical match to determine a root cause with confidence. "
                "The closest past incidents are weak matches; manual investigation is "
                "recommended before acting."
            ),
            "evidence": detector_evidence + [
                f"best retrieval similarity {rconf:.2f} is below the "
                f"{config.RAG_LOW_RETRIEVAL_CUTOFF:.2f} confidence cutoff"
            ],
            "citations": citations,
            "confidence": round(min(0.35, 0.15 + rconf), 4),
            "low_confidence": True,
        }

    top = retrieved[0]
    return {
        "hypothesis": (
            f"Likely root cause resembles {top['id']} ({top['title']}): "
            f"{top['root_cause']} Consider the resolution: {top['resolution']}"
        ),
        "evidence": detector_evidence + [
            f"closest historical incident {top['id']} similarity {top['similarity']:.2f}"
        ],
        "citations": citations,
        "confidence": round(min(0.9, 0.5 + 0.4 * rconf), 4),
        "low_confidence": False,
    }


def _offline_plan(payload: dict) -> dict:
    a = payload["anomaly"]
    rca = payload.get("rca", {})
    template = _PLAN_TEMPLATES.get(a["metric"], _DEFAULT_TEMPLATE)

    steps = []
    order = 1
    if rca.get("low_confidence"):
        steps.append({
            "order": order,
            "action": "Manually investigate: root cause is uncertain (low retrieval confidence)",
            "risk": "low", "customer_facing": False, "destructive": False,
            "rationale": "RCA confidence is low; verify before taking corrective action.",
        })
        order += 1
    for action, risk, cf, destr in template:
        steps.append({
            "order": order, "action": action, "risk": risk,
            "customer_facing": cf, "destructive": destr,
            "rationale": f"Addresses {a['metric']} anomaly on {a['source']}.",
        })
        order += 1

    overall = max((s["risk"] for s in steps), key=lambda r: _RISK_ORDER[r])
    rca_conf = float(rca.get("confidence", 0.5))
    return {
        "summary": f"Action plan for {a['metric']} anomaly on {a['source']} "
                   f"(severity {a['severity']}).",
        "steps": steps,
        "overall_risk": overall,
        "confidence": round(rca_conf * 0.95, 4),
    }


_OFFLINE_DISPATCH: dict[str, Callable[[dict], dict]] = {
    TASK_TRIAGE: _offline_triage,
    TASK_RCA: _offline_rca,
    TASK_PLAN: _offline_plan,
}


# ===========================================================================
# Anthropic backend prompts
# ===========================================================================
_SYSTEM = (
    "You are OpsCortex, an operations-intelligence analyst. You are given "
    "deterministic anomaly facts and retrieved historical incidents. You only "
    "explain and plan; you never recompute the numbers. Respond with a single "
    "JSON object and nothing else. If the retrieved incidents are weak matches, "
    "say so and set low_confidence to true rather than inventing a cause."
)

_PROMPTS = {
    TASK_TRIAGE: (
        "Classify this anomaly.\n{payload}\n\n"
        'Return JSON: {{"severity": "LOW|MEDIUM|HIGH|CRITICAL", '
        '"priority": "P0|P1|P2|P3", "rationale": "..."}}'
    ),
    TASK_RCA: (
        "Perform root-cause analysis grounded in the retrieved incidents.\n{payload}\n\n"
        'Return JSON: {{"hypothesis": "...", "evidence": ["..."], '
        '"citations": [{{"incident_id": "...", "title": "...", "similarity": 0.0}}], '
        '"confidence": 0.0, "low_confidence": false}}'
    ),
    TASK_PLAN: (
        "Draft a prioritized action plan.\n{payload}\n\n"
        'Return JSON: {{"summary": "...", "steps": [{{"order": 1, "action": "...", '
        '"risk": "low|medium|high", "customer_facing": false, "destructive": false, '
        '"rationale": "..."}}], "overall_risk": "low|medium|high", "confidence": 0.0}}'
    ),
}


# ===========================================================================
# LLM client
# ===========================================================================
class LLMClient:
    def __init__(self, backend: Optional[str] = None):
        requested = (backend or config.LLM_BACKEND)
        if requested == "anthropic" and not config.ANTHROPIC_API_KEY:
            log.warning("OPSCORTEX_LLM_BACKEND=anthropic but ANTHROPIC_API_KEY is "
                        "not set; falling back to the offline backend.")
            requested = "offline"
        self.backend = requested
        self._chat_cache: dict[str, Any] = {}
        log.info("LLM backend: %s", self.backend)

    # -- public API --------------------------------------------------------
    def structured(self, task: str, payload: dict, model: str,
                   node: Optional[str] = None) -> tuple[dict, dict]:
        node = node or task
        payload, truncated = enforce_budget(node, payload)
        prompt_tokens = estimate_tokens(json.dumps(payload, default=str))

        if self.backend == "offline":
            data = _OFFLINE_DISPATCH[task](payload)
        else:
            data = self._anthropic_structured(task, payload, model)

        meta = {
            "node": node,
            "model_used": f"{model} ({self.backend})",
            "backend": self.backend,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": estimate_tokens(json.dumps(data, default=str)),
            "truncated": truncated,
        }
        return data, meta

    # -- anthropic path ----------------------------------------------------
    def _get_chat(self, model: str):
        if model not in self._chat_cache:
            from langchain_anthropic import ChatAnthropic  # lazy import
            self._chat_cache[model] = ChatAnthropic(
                model=model,
                api_key=config.ANTHROPIC_API_KEY,
                timeout=config.LLM_REQUEST_TIMEOUT_SECONDS,
                max_retries=0,  # we handle retries via tenacity
            )
        return self._chat_cache[model]

    def _anthropic_structured(self, task: str, payload: dict, model: str) -> dict:
        user = _PROMPTS[task].format(payload=json.dumps(payload, default=str, indent=2))

        @tenacity.retry(
            stop=tenacity.stop_after_attempt(config.LLM_MAX_RETRIES),
            wait=tenacity.wait_exponential(multiplier=config.LLM_BACKOFF_BASE_SECONDS,
                                           max=config.LLM_BACKOFF_MAX_SECONDS),
            retry=tenacity.retry_if_exception_type(Exception),
            reraise=True,
        )
        def _invoke() -> str:
            chat = self._get_chat(model)
            resp = chat.invoke([("system", _SYSTEM), ("user", user)])
            content = resp.content
            if isinstance(content, list):  # content blocks
                content = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in content
                )
            return content

        raw = _invoke()
        return parse_json(raw)


# Module-level singleton.
_CLIENT: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient()
    return _CLIENT


def reset_llm(backend: Optional[str] = None) -> LLMClient:
    """Re-create the client (used by tests / backend switches)."""
    global _CLIENT
    _CLIENT = LLMClient(backend=backend)
    return _CLIENT


# ===========================================================================
# Append-only audit trail
# ===========================================================================
def record_audit(step: str, status: str, inputs: str = "", outputs: str = "",
                 model_used: Optional[str] = None) -> AuditEntry:
    """Create an AuditEntry and append it to the JSONL audit file. The in-memory
    copy is also accumulated in graph state via the additive `audit_log` channel."""
    entry = AuditEntry(step=step, status=status, inputs=inputs[:2000],
                       outputs=outputs[:2000], model_used=model_used)
    try:
        with open(config.AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")
    except OSError as exc:  # never let audit IO crash the pipeline
        log.warning("could not append to audit file %s: %s", config.AUDIT_LOG_PATH, exc)
    return entry


# ===========================================================================
# Human-in-the-loop approval (non-TTY)
# ===========================================================================
class ApprovalProvider(ABC):
    """Pluggable approval interface. Implementations must NOT depend on a TTY —
    wire these to Slack/email/a dashboard in production."""

    @abstractmethod
    def request_approval(self, plan: ActionPlan, context: dict) -> ApprovalDecision:
        ...


class QueueApprovalProvider(ApprovalProvider):
    """Default: queue the request for an external approver (status=PENDING).
    Nothing risky is executed automatically."""

    def request_approval(self, plan: ActionPlan, context: dict) -> ApprovalDecision:
        log.info("HITL: plan %s queued for human approval (%s)", plan.id,
                 ", ".join(plan.approval_reasons) or "policy")
        return ApprovalDecision(plan_id=plan.id, anomaly_id=plan.anomaly_id,
                                status=ApprovalStatus.PENDING, approver="queue",
                                reason="Queued for external approver (Slack/email/dashboard).")


class AutoApproveProvider(ApprovalProvider):
    """Approves everything — for tests/demos only."""

    def __init__(self, approver: str = "test-auto-approver"):
        self.approver = approver

    def request_approval(self, plan: ActionPlan, context: dict) -> ApprovalDecision:
        return ApprovalDecision(plan_id=plan.id, anomaly_id=plan.anomaly_id,
                                status=ApprovalStatus.APPROVED, approver=self.approver,
                                reason="Auto-approved (test/demo provider).")


class DenyApproveProvider(ApprovalProvider):
    """Denies everything — for tests."""

    def request_approval(self, plan: ActionPlan, context: dict) -> ApprovalDecision:
        return ApprovalDecision(plan_id=plan.id, anomaly_id=plan.anomaly_id,
                                status=ApprovalStatus.DENIED, approver="test-deny",
                                reason="Denied (test provider).")


class CallbackApprovalProvider(ApprovalProvider):
    """Delegates to a callable that returns True/False/ApprovalDecision."""

    def __init__(self, fn: Callable[[ActionPlan, dict], Any], approver: str = "callback"):
        self.fn = fn
        self.approver = approver

    def request_approval(self, plan: ActionPlan, context: dict) -> ApprovalDecision:
        result = self.fn(plan, context)
        if isinstance(result, ApprovalDecision):
            return result
        status = ApprovalStatus.APPROVED if result else ApprovalStatus.DENIED
        return ApprovalDecision(plan_id=plan.id, anomaly_id=plan.anomaly_id,
                                status=status, approver=self.approver,
                                reason=f"Callback returned {result!r}.")


_APPROVAL_PROVIDER: ApprovalProvider = QueueApprovalProvider()


def set_approval_provider(provider: ApprovalProvider) -> None:
    global _APPROVAL_PROVIDER
    _APPROVAL_PROVIDER = provider


def get_approval_provider() -> ApprovalProvider:
    return _APPROVAL_PROVIDER
