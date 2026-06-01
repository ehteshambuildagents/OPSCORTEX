"""
main.py — Run OpsCortex end-to-end for one scenario and print the run summary
and full audit trail.

Usage:
    python main.py [scenario]

Scenarios: default, payment_failure_spike, ticket_surge, latency_creep,
           noisy_benign, low_retrieval_confidence, all_normal
"""
from __future__ import annotations

import json
import sys

import config
from graph import build_graph, run
from nodes import build_summary


def _print_audit_trail(state) -> None:
    print("\n=== AUDIT TRAIL (append-only) ===")
    for i, e in enumerate(state.get("audit_log", []), 1):
        model = f" | model={e.model_used}" if e.model_used else ""
        ts = e.timestamp.strftime("%H:%M:%S")
        print(f"{i:>2}. [{ts}] {e.step:<12} {e.status:<22}{model}")
        if e.inputs:
            print(f"      in : {e.inputs}")
        if e.outputs:
            print(f"      out: {e.outputs}")


def _print_details(state) -> None:
    if state.get("anomalies"):
        print("\n=== ANOMALIES ===")
        for a in state["anomalies"]:
            print(f"- [{a.severity.value}] {a.metric}@{a.source} "
                  f"obs={a.observed_value:g} base~{a.baseline:g} "
                  f"score={a.severity_score:.2f} via {','.join(a.detectors)}")
    for rc in state.get("rca", []):
        flag = " (LOW CONFIDENCE)" if rc.low_confidence else ""
        print(f"\n=== ROOT CAUSE [{rc.anomaly_id}]{flag} conf={rc.confidence:.2f} ===")
        print(f"  {rc.hypothesis}")
        if rc.citations:
            print("  citations: " + "; ".join(
                f"{c.incident_id}({c.similarity:.2f})" for c in rc.citations))
    for p in state.get("plans", []):
        tag = "HITL" if p.requires_approval else "AUTO"
        print(f"\n=== PLAN {p.id} [{tag}] risk={p.overall_risk.value} conf={p.confidence:.2f} ===")
        if p.approval_reasons:
            print(f"  approval needed: {', '.join(p.approval_reasons)}")
        for s in p.steps:
            flags = []
            if s.destructive:
                flags.append("destructive")
            if s.customer_facing:
                flags.append("customer-facing")
            fl = f" [{', '.join(flags)}]" if flags else ""
            print(f"  {s.order}. ({s.risk.value}) {s.action}{fl}")


def main() -> int:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "default"
    config.setup_logging()
    app = build_graph()
    state = run(scenario, thread_id=f"cli-{scenario}", app=app)

    _print_details(state)
    _print_audit_trail(state)

    print("\n=== END-OF-RUN SUMMARY ===")
    print(json.dumps(build_summary(state), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
