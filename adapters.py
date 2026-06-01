"""
adapters.py — Typed data-source adapters (mock now, swappable later).

Each data source is a subclass of `DataSourceAdapter`. To connect a real API
(Stripe, Zendesk, Datadog, a status page), replace the body of the matching
adapter's `fetch()` method with a real client call that returns `Signal`
objects — nothing else in the codebase needs to change.

Mock feeds are generated deterministically from a seed so runs and tests are
reproducible. Anomalies are injected per named "profile" so the test suite can
exercise specific scenarios.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

from state import Signal, SignalPoint

POINTS = 48                      # hourly points (~2 days of history)
_BASE_TS = datetime(2026, 5, 29, 0, 0, tzinfo=timezone.utc)


def _series(metric: str, unit: str, source: str, seed: int,
            mean: float, noise: float,
            inject: dict | None = None) -> Signal:
    """Build one deterministic metric series with optional injected anomaly.

    `inject` shapes the tail of the series:
      {"kind": "spike"|"drop"|"creep", "to": <value>, "span": <n points>}
    A "spike"/"drop" sets the last `span` points to ~`to`; "creep" ramps the
    last `span` points from `mean` toward `to`.
    """
    rng = random.Random(seed)
    pts: list[SignalPoint] = []
    for i in range(POINTS):
        val = rng.gauss(mean, noise)
        pts.append(SignalPoint(timestamp=_BASE_TS + timedelta(hours=i), value=val))

    if inject:
        kind = inject["kind"]
        span = inject.get("span", 3)
        target = inject["to"]
        if kind in ("spike", "drop"):
            for j in range(POINTS - span, POINTS):
                pts[j].value = rng.gauss(target, noise)
        elif kind == "creep":
            for k, j in enumerate(range(POINTS - span, POINTS)):
                frac = (k + 1) / span
                pts[j].value = mean + (target - mean) * frac + rng.gauss(0, noise * 0.3)

    # Clamp to sane physical ranges per unit.
    for p in pts:
        if unit == "%":
            p.value = max(0.0, p.value)
        p.value = round(p.value, 3)

    return Signal(source=source, metric=metric, unit=unit, points=pts,
                  metadata={"seed": seed, "injected": bool(inject)})


def _flatten_tail(sig: Signal, value: float, span: int = 3) -> Signal:
    """Force the tail back to a calm baseline (used for benign-noise profiles to
    guarantee the latest point is not an outlier)."""
    for j in range(len(sig.points) - span, len(sig.points)):
        sig.points[j].value = round(value, 3)
    return sig


class DataSourceAdapter(ABC):
    """Interface every data source implements. Swap `fetch()` for a real API."""
    name: str = "abstract"

    def __init__(self, profile: str = "normal", seed: int = 42):
        self.profile = profile
        self.seed = seed

    @abstractmethod
    def fetch(self) -> list[Signal]:
        """Return the current metric series for this source."""
        raise NotImplementedError


class PaymentEventsAdapter(DataSourceAdapter):
    """Stripe-like payment events. Real swap: query the Stripe API for charge
    outcomes and aggregate into failure rate / volume series."""
    name = "payment_events"

    def fetch(self) -> list[Signal]:
        inj = {"kind": "spike", "to": 28.0, "span": 3} if self.profile == "payment_spike" else None
        fail = _series("payment_failure_rate", "%", self.name, self.seed + 1,
                       mean=2.0, noise=0.5, inject=inj)
        vol = _series("payment_volume", "count", self.name, self.seed + 2,
                      mean=1000, noise=40)
        return [fail, vol]


class SupportTicketsAdapter(DataSourceAdapter):
    """Zendesk-like support tickets. Real swap: query the Zendesk API for ticket
    counts grouped by interval."""
    name = "support_tickets"

    def fetch(self) -> list[Signal]:
        inj = {"kind": "spike", "to": 85.0, "span": 4} if self.profile == "ticket_surge" else None
        tickets = _series("ticket_volume", "count/hr", self.name, self.seed + 3,
                          mean=20.0, noise=3.0, inject=inj)
        return [tickets]


class AppMetricsAdapter(DataSourceAdapter):
    """APM-like application metrics: latency, errors, churn, and recommendation
    CTR. Real swap: query Datadog/Prometheus/New Relic."""
    name = "app_metrics"

    def fetch(self) -> list[Signal]:
        lat_inj = {"kind": "creep", "to": 250.0, "span": 14} if self.profile == "latency_creep" else None
        latency = _series("p95_latency_ms", "ms", self.name, self.seed + 4,
                          mean=200.0, noise=8.0, inject=lat_inj)

        if self.profile == "noisy_benign":
            # High-variance but harmless error-rate jitter; tail flattened so the
            # latest point is never an outlier and no detector should fire.
            error = _series("error_rate", "%", self.name, self.seed + 5,
                            mean=1.0, noise=0.35)
            error = _flatten_tail(error, value=1.05, span=3)
        else:
            error = _series("error_rate", "%", self.name, self.seed + 5,
                            mean=0.8, noise=0.2)

        churn = _series("churn_rate", "%", self.name, self.seed + 6, mean=1.0, noise=0.1)

        signals = [latency, error, churn]

        # reco_ctr only participates in the low-retrieval-confidence scenario:
        # a novel failure mode (recommendation model) absent from the incident
        # corpus, so RCA must admit uncertainty.
        if self.profile == "reco_collapse":
            reco = _series("reco_ctr", "%", self.name, self.seed + 7,
                           mean=5.0, noise=0.3, inject={"kind": "drop", "to": 1.0, "span": 4})
            signals.append(reco)
        return signals


class InfraAlertsAdapter(DataSourceAdapter):
    """Uptime/infra alerts. Real swap: query your status page / Pingdom /
    health-check API for uptime per interval."""
    name = "infra_alerts"

    def fetch(self) -> list[Signal]:
        inj = {"kind": "drop", "to": 98.5, "span": 3} if self.profile == "infra_drop" else None
        uptime = _series("uptime_pct", "%", self.name, self.seed + 8,
                         mean=99.96, noise=0.02, inject=inj)
        return [uptime]


# Map each scenario to a per-source profile. A source not listed runs "normal".
SCENARIO_PROFILES: dict[str, dict[str, str]] = {
    "default": {"payment_events": "payment_spike"},
    "payment_failure_spike": {"payment_events": "payment_spike"},
    "ticket_surge": {"support_tickets": "ticket_surge"},
    "latency_creep": {"app_metrics": "latency_creep"},
    "noisy_benign": {"app_metrics": "noisy_benign"},
    "low_retrieval_confidence": {"app_metrics": "reco_collapse"},
    "all_normal": {},
}

_ADAPTER_CLASSES = [
    PaymentEventsAdapter,
    SupportTicketsAdapter,
    AppMetricsAdapter,
    InfraAlertsAdapter,
]


def build_adapters(scenario: str = "default", seed: int = 42) -> list[DataSourceAdapter]:
    """Instantiate the four adapters with the profiles for a given scenario."""
    profiles = SCENARIO_PROFILES.get(scenario, {})
    return [cls(profile=profiles.get(cls.name, "normal"), seed=seed)
            for cls in _ADAPTER_CLASSES]


def fetch_all(scenario: str = "default", seed: int = 42) -> list[Signal]:
    """Pull and concatenate signals from every data source for a scenario."""
    signals: list[Signal] = []
    for adapter in build_adapters(scenario, seed):
        signals.extend(adapter.fetch())
    return signals
