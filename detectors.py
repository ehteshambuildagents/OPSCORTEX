"""
detectors.py — Deterministic anomaly detection (NO LLM).

Four detectors: absolute threshold breach, rolling-average z-score deviation,
sudden percent-change, and trend reversal. Each inspects the *current* state of
a series and returns a raw finding. The orchestrator combines findings per
metric, applies the metric weight, and emits typed `Anomaly` objects with a
numeric severity. The LLM never does this math.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Optional

import config
from state import Anomaly, Severity, Signal


@dataclass
class Finding:
    detector: str
    observed: float
    baseline: float
    deviation: float          # raw, normalized 0..1 (pre metric-weight)
    direction: str
    evidence: dict


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def threshold_detector(sig: Signal, cfg: dict) -> Optional[Finding]:
    val = sig.latest()
    warn, crit, direction = cfg["warn"], cfg["critical"], cfg["direction"]
    breached = val >= warn if direction == "up" else val <= warn
    if not breached:
        return None
    denom = (crit - warn) if direction == "up" else (warn - crit)
    frac = (val - warn) / denom if direction == "up" else (warn - val) / denom
    frac = _clamp01(frac)
    deviation = _clamp01(0.25 + 0.75 * frac)
    return Finding("threshold", val, warn, deviation, direction,
                   {"warn": warn, "critical": crit, "value": val})


def zscore_detector(sig: Signal, cfg: dict) -> Optional[Finding]:
    vals = sig.values()
    gap = max(1, config.ZSCORE_GAP)
    if len(vals) < gap + 3:
        return None
    # Baseline = a trailing window that ENDS `gap` points before now, so a
    # multi-point spike in the recent tail does not contaminate its own baseline.
    window = vals[-(config.ZSCORE_WINDOW + gap):-gap]
    if len(window) < 3:
        return None
    mean = statistics.fmean(window)
    std = statistics.pstdev(window)
    latest = vals[-1]
    if std < 1e-9:
        return None
    z = (latest - mean) / std
    directional_z = z if cfg["direction"] == "up" else -z
    if directional_z < config.ZSCORE_THRESHOLD:
        return None
    deviation = _clamp01(abs(directional_z) / config.ZSCORE_SATURATION)
    return Finding("zscore", latest, mean, deviation, cfg["direction"],
                   {"z": round(z, 2), "mean": round(mean, 3), "std": round(std, 3),
                    "window": len(window)})


def pct_change_detector(sig: Signal, cfg: dict) -> Optional[Finding]:
    vals = sig.values()
    if len(vals) < 2:
        return None
    prev, cur = vals[-2], vals[-1]
    floor = cfg.get("floor", 0.0)
    if abs(prev) < floor and abs(cur) < floor:
        return None
    if abs(prev) < 1e-9:
        return None
    pct = (cur - prev) / abs(prev)
    directional = pct if cfg["direction"] == "up" else -pct
    if directional < config.PCT_CHANGE_THRESHOLD:
        return None
    deviation = _clamp01(abs(directional) / config.PCT_CHANGE_SATURATION)
    return Finding("pct_change", cur, prev, deviation, cfg["direction"],
                   {"pct_change": round(pct, 3), "prev": prev, "cur": cur})


def trend_reversal_detector(sig: Signal, cfg: dict) -> Optional[Finding]:
    vals = sig.values()
    w = config.TREND_WINDOW
    if len(vals) < 2 * w:
        return None
    older = vals[-2 * w:-w]
    recent = vals[-w:]
    slope_old = older[-1] - older[0]
    slope_new = recent[-1] - recent[0]
    flipped = (slope_old <= 0 < slope_new) or (slope_old >= 0 > slope_new)
    bad_dir = slope_new > 0 if cfg["direction"] == "up" else slope_new < 0
    if not (flipped and bad_dir and abs(slope_new) > 1e-6):
        return None
    return Finding("trend_reversal", recent[-1], older[0],
                   config.TREND_REVERSAL_DEVIATION, cfg["direction"],
                   {"slope_old": round(slope_old, 3), "slope_new": round(slope_new, 3)})


_DETECTORS = {
    "threshold": threshold_detector,
    "zscore": zscore_detector,
    "pct_change": pct_change_detector,
    "trend_reversal": trend_reversal_detector,
}


def detect_signal(sig: Signal) -> Optional[Anomaly]:
    """Run all configured detectors for one signal and fold them into a single
    typed Anomaly (or None if nothing fired)."""
    cfg = config.DETECTOR_CONFIG.get(sig.metric)
    if not cfg:
        return None

    findings: list[Finding] = []
    for name in cfg["detectors"]:
        fn = _DETECTORS[name]
        f = fn(sig, cfg)
        if f is not None:
            findings.append(f)
    if not findings:
        return None

    top = max(findings, key=lambda f: f.deviation)
    weight = cfg["weight"]
    severity_score = _clamp01(top.deviation * weight)
    label = config.severity_label(severity_score)

    phrase = config.METRIC_PHRASES.get(sig.metric, sig.metric)
    description = (
        f"{phrase}: {sig.metric} on {sig.source} is {top.observed:g} "
        f"(baseline ~{top.baseline:g}, direction {cfg['direction']}). "
        f"Detected by: {', '.join(sorted({f.detector for f in findings}))}."
    )

    return Anomaly(
        source=sig.source,
        metric=sig.metric,
        detectors=sorted({f.detector for f in findings}),
        description=description,
        direction=cfg["direction"],
        observed_value=round(top.observed, 4),
        baseline=round(top.baseline, 4),
        deviation=round(top.deviation, 4),
        severity_score=round(severity_score, 4),
        severity=Severity(label),
        timestamp=sig.latest_ts(),
        evidence={"weight": weight,
                  "findings": [f.__dict__ for f in findings]},
    )


def run_detectors(signals: list[Signal]) -> list[Anomaly]:
    """Detect anomalies across all signals, sorted by severity (desc)."""
    anomalies = [a for a in (detect_signal(s) for s in signals) if a is not None]
    anomalies.sort(key=lambda a: a.severity_score, reverse=True)
    return anomalies
