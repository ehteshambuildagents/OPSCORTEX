"""
state.py — Pydantic data models and the LangGraph state schema.

Every value that flows between nodes is a validated Pydantic model. The graph
state itself is a TypedDict; the additive channels use an additive reducer so
each node appends rather than overwrites.
"""
from __future__ import annotations

import operator
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ApprovalStatus(str, Enum):
    APPROVED = "approved"
    DENIED = "denied"
    PENDING = "pending"


# ---------------------------------------------------------------------------
# Signals (raw inputs from data sources)
# ---------------------------------------------------------------------------
class SignalPoint(BaseModel):
    timestamp: datetime
    value: float


class Signal(BaseModel):
    """A single metric time series from one data source."""
    source: str
    metric: str
    unit: str = ""
    points: list[SignalPoint] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def values(self) -> list[float]:
        return [p.value for p in self.points]

    def latest(self) -> float:
        return self.points[-1].value

    def latest_ts(self) -> datetime:
        return self.points[-1].timestamp


# ---------------------------------------------------------------------------
# Anomalies (output of deterministic detectors)
# ---------------------------------------------------------------------------
class Anomaly(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("anom"))
    source: str
    metric: str
    detectors: list[str]
    description: str
    direction: str
    observed_value: float
    baseline: float
    deviation: float = Field(ge=0.0, le=1.0)
    severity_score: float = Field(ge=0.0, le=1.0)
    severity: Severity
    timestamp: datetime = Field(default_factory=_utcnow)
    evidence: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------
class TriageResult(BaseModel):
    anomaly_id: str
    severity: Severity
    priority: str
    rationale: str
    model_used: str


# ---------------------------------------------------------------------------
# Root-cause analysis (RAG-grounded)
# ---------------------------------------------------------------------------
class Citation(BaseModel):
    incident_id: str
    title: str
    similarity: float = Field(ge=0.0, le=1.0)


class RootCauseAnalysis(BaseModel):
    anomaly_id: str
    hypothesis: str
    evidence: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    low_confidence: bool = False
    retrieval_confidence: float = Field(ge=0.0, le=1.0)
    model_used: str


# ---------------------------------------------------------------------------
# Action plan
# ---------------------------------------------------------------------------
class ActionStep(BaseModel):
    order: int
    action: str
    risk: RiskLevel
    customer_facing: bool = False
    destructive: bool = False
    rationale: str = ""


class ActionPlan(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("plan"))
    anomaly_id: str
    summary: str
    steps: list[ActionStep] = Field(default_factory=list)
    overall_risk: RiskLevel
    confidence: float = Field(ge=0.0, le=1.0)
    requires_approval: bool = False
    approval_reasons: list[str] = Field(default_factory=list)
    model_used: str

    @field_validator("steps")
    @classmethod
    def _non_empty(cls, v: list[ActionStep]) -> list[ActionStep]:
        if not v:
            raise ValueError("an action plan must contain at least one step")
        return v


# ---------------------------------------------------------------------------
# Approval (HITL)
# ---------------------------------------------------------------------------
class ApprovalDecision(BaseModel):
    plan_id: str
    anomaly_id: str
    status: ApprovalStatus
    approver: str
    reason: str = ""
    timestamp: datetime = Field(default_factory=_utcnow)

    @property
    def approved(self) -> bool:
        return self.status == ApprovalStatus.APPROVED


# ---------------------------------------------------------------------------
# Audit trail (append-only)
# ---------------------------------------------------------------------------
class AuditEntry(BaseModel):
    step: str
    status: str
    model_used: Optional[str] = None
    inputs: str = ""
    outputs: str = ""
    timestamp: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------
class OpsState(TypedDict, total=False):
    scenario: str
    thread_id: str
    signals: list[Signal]
    anomalies: list[Anomaly]
    triage: list[TriageResult]
    rca: list[RootCauseAnalysis]
    plans: list[ActionPlan]
    approvals: Annotated[list[ApprovalDecision], operator.add]
    executed_actions: Annotated[list[dict], operator.add]
    auto_plan_ids: list[str]
    hitl_plan_ids: list[str]
    audit_log: Annotated[list[AuditEntry], operator.add]
    token_usage: Annotated[list[dict], operator.add]
    error: Optional[dict]
    summary: dict
