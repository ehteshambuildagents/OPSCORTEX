"""
config.py — Central configuration for OpsCortex.

Every tunable knob lives here: model names, the LLM backend toggle, detector
thresholds, routing/governance rules, confidence cutoffs, token budgets, retry
policy, and logging. Nothing operational should be hardcoded deeper in the code.
"""
from __future__ import annotations

import logging
import os
import sys

# ---------------------------------------------------------------------------
# Models  (configurable constants — never hardcode these deep in the code)
# ---------------------------------------------------------------------------
SYNTHESIS_MODEL = "claude-sonnet-4-6"
TRIAGE_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------
# "offline"  -> deterministic, rule-based backend. No API key, no network, no
#               cost. This is the default so the graph and tests run green
#               anywhere. It produces schema-valid structured output derived
#               from the deterministic detector facts.
# "anthropic" -> real langchain-anthropic calls. Requires ANTHROPIC_API_KEY.
LLM_BACKEND = os.environ.get("OPSCORTEX_LLM_BACKEND", "offline").strip().lower()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()

# ---------------------------------------------------------------------------
# Reliability: retry/backoff for LLM calls
# ---------------------------------------------------------------------------
LLM_MAX_RETRIES = 3
LLM_BACKOFF_BASE_SECONDS = 0.5
LLM_BACKOFF_MAX_SECONDS = 8.0
LLM_REQUEST_TIMEOUT_SECONDS = 60

# ---------------------------------------------------------------------------
# Token budgeting (rough char/4 estimate; per-node ceilings on prompt size)
# ---------------------------------------------------------------------------
TOKEN_BUDGETS = {
    "triage": 1500,
    "rca": 4000,
    "plan": 3000,
}
CHARS_PER_TOKEN = 4

# ---------------------------------------------------------------------------
# Anomaly detection (deterministic math only — the LLM never computes these)
# ---------------------------------------------------------------------------
ZSCORE_WINDOW = 24
ZSCORE_GAP = 4              # guard gap: ignore the most recent N points when
                            # computing the baseline, so a multi-point spike does
                            # not contaminate its own baseline
ZSCORE_THRESHOLD = 3.0
ZSCORE_SATURATION = 8.0
PCT_CHANGE_THRESHOLD = 0.50
PCT_CHANGE_SATURATION = 2.0
TREND_REVERSAL_DEVIATION = 0.30
TREND_WINDOW = 6

# Per-metric detector configuration.
#   weight    : metric importance multiplier applied to the raw deviation
#   direction : "up" (high is bad) or "down" (low is bad) for threshold checks
#   warn/crit : threshold band used by the threshold detector
#   floor     : pct_change is ignored when BOTH consecutive values are below this
#   detectors : which detectors apply to this metric
DETECTOR_CONFIG = {
    "payment_failure_rate": {"weight": 1.00, "direction": "up", "warn": 10.0, "critical": 20.0,
                             "floor": 8.0, "detectors": ["threshold", "zscore", "pct_change"]},
    "payment_volume":       {"weight": 0.70, "direction": "down", "warn": 700, "critical": 500,
                             "floor": 600, "detectors": ["zscore"]},
    "ticket_volume":        {"weight": 0.45, "direction": "up", "warn": 60, "critical": 120,
                             "floor": 40, "detectors": ["zscore", "pct_change"]},
    "p95_latency_ms":       {"weight": 0.50, "direction": "up", "warn": 240, "critical": 400,
                             "floor": 230, "detectors": ["threshold", "zscore", "trend_reversal"]},
    "error_rate":           {"weight": 0.90, "direction": "up", "warn": 3.0, "critical": 8.0,
                             "floor": 2.5, "detectors": ["threshold", "zscore", "pct_change"]},
    "churn_rate":           {"weight": 0.70, "direction": "up", "warn": 3.0, "critical": 6.0,
                             "floor": 2.5, "detectors": ["zscore"]},
    "uptime_pct":           {"weight": 1.00, "direction": "down", "warn": 99.9, "critical": 99.0,
                             "floor": 0.0, "detectors": ["threshold", "zscore"]},
    "reco_ctr":             {"weight": 0.70, "direction": "down", "warn": 3.0, "critical": 1.5,
                             "floor": 0.5, "detectors": ["threshold", "pct_change"]},
}

METRIC_PHRASES = {
    "payment_failure_rate": "payment failure rate spike (declined charges / processor failures)",
    "payment_volume": "payment volume drop",
    "ticket_volume": "support ticket surge",
    "p95_latency_ms": "API p95 latency increase",
    "error_rate": "application error rate increase",
    "churn_rate": "customer churn rate increase",
    "uptime_pct": "infrastructure uptime drop / outage",
    "reco_ctr": "recommendation model click-through collapse",
}

# Map raw severity score (0..1) -> label. Checked low to high.
SEVERITY_CUTOFFS = [
    (0.20, "LOW"),
    (0.50, "MEDIUM"),
    (0.75, "HIGH"),
    (1.01, "CRITICAL"),
]

# ---------------------------------------------------------------------------
# RAG / root-cause confidence
# ---------------------------------------------------------------------------
RAG_TOP_K = 3
# Valid metric anomalies retrieve at ~0.55-0.61 similarity against the corpus;
# a genuinely novel failure mode lands ~0.25. 0.35 separates them with margin.
RAG_LOW_RETRIEVAL_CUTOFF = 0.35

# ---------------------------------------------------------------------------
# Governance / routing  (auto-action vs human-in-the-loop)
# ---------------------------------------------------------------------------
HITL_SEVERITIES = {"HIGH", "CRITICAL"}
RCA_CONFIDENCE_CUTOFF = 0.45
PLAN_CONFIDENCE_CUTOFF = 0.45
HITL_RISK_LEVELS = {"high"}

# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------
AUDIT_LOG_PATH = os.environ.get("OPSCORTEX_AUDIT_LOG", "audit_log.jsonl")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("OPSCORTEX_LOG_LEVEL", "INFO").upper()


def setup_logging() -> None:
    """Configure structured-ish stdout logging. Idempotent."""
    root = logging.getLogger("opscortex")
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(handler)
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"opscortex.{name}")


def severity_label(score: float) -> str:
    """Map a 0..1 severity score to a label using SEVERITY_CUTOFFS."""
    for cutoff, label in SEVERITY_CUTOFFS:
        if score < cutoff:
            return label
    return "CRITICAL"
