"""
test_runner.py — Labeled-fixture test suite + accuracy report for OpsCortex.

Runs the five required scenarios against the compiled graph (offline backend,
deterministic) and checks each against its label. Then it computes and prints an
accuracy report: pipeline success rate, anomaly precision/recall, false-positive
rate, and how often the agent correctly admits low confidence instead of
inventing a cause.

Exit code 0 iff every scenario passes.
"""
from __future__ import annotations

import logging

import tools
from graph import build_graph, run
from nodes import build_summary

logging.getLogger("opscortex").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Labeled fixtures
# ---------------------------------------------------------------------------
FIXTURES = [
    {"name": "1. Payment failure spike (high -> HITL)",
     "scenario": "payment_failure_spike", "should_raise": True,
     "metric": "payment_failure_rate", "severity": {"HIGH", "CRITICAL"},
     "route": "hitl", "low_confidence": False},
    {"name": "2. Support ticket surge (medium -> action plan)",
     "scenario": "ticket_surge", "should_raise": True,
     "metric": "ticket_volume", "severity": {"MEDIUM"},
     "route": "auto", "low_confidence": False},
    {"name": "3. Slow latency creep (low -> auto monitor)",
     "scenario": "latency_creep", "should_raise": True,
     "metric": "p95_latency_ms", "severity": {"LOW"},
     "route": "auto", "low_confidence": False},
    {"name": "4. Noisy-but-benign fluctuation (no anomaly)",
     "scenario": "noisy_benign", "should_raise": False,
     "metric": None, "severity": set(),
     "route": "none", "low_confidence": False},
    {"name": "5. Low-retrieval-confidence incident (admit uncertainty)",
     "scenario": "low_retrieval_confidence", "should_raise": True,
     "metric": "reco_ctr", "severity": {"LOW", "MEDIUM", "HIGH", "CRITICAL"},
     "route": "hitl", "low_confidence": True},
]


def _actual_route(summary: dict) -> str:
    if summary.get("hitl_actions", 0) > 0:
        return "hitl"
    if summary.get("auto_actions", 0) > 0:
        return "auto"
    return "none"


def _check(fixture: dict, state: dict, summary: dict) -> tuple[bool, list[str]]:
    fails: list[str] = []
    anomalies = state.get("anomalies", [])
    raised = len(anomalies) > 0

    if summary.get("status") != "ok":
        fails.append(f"pipeline status={summary.get('status')}")

    if fixture["should_raise"] != raised:
        fails.append(f"raised={raised} expected={fixture['should_raise']}")

    if fixture["should_raise"] and raised:
        metrics = {a.metric for a in anomalies}
        if fixture["metric"] not in metrics:
            fails.append(f"expected metric {fixture['metric']} not in {metrics}")
        target = next((a for a in anomalies if a.metric == fixture["metric"]), None)
        if target and fixture["severity"] and target.severity.value not in fixture["severity"]:
            fails.append(f"severity {target.severity.value} not in {fixture['severity']}")

    actual_route = _actual_route(summary)
    if actual_route != fixture["route"]:
        fails.append(f"route={actual_route} expected={fixture['route']}")

    admitted = summary.get("low_confidence_rca", 0) > 0
    if admitted != fixture["low_confidence"]:
        fails.append(f"low_confidence_admitted={admitted} expected={fixture['low_confidence']}")

    return (not fails), fails


def main() -> int:
    tools.reset_llm("offline")
    tools.set_approval_provider(tools.QueueApprovalProvider())
    app = build_graph()

    results = []
    tp = fp = tn = fn = 0
    lc_correct = lc_total = 0
    pipeline_ok = 0

    print("=" * 72)
    print("OpsCortex - labeled-fixture test suite")
    print("=" * 72)

    for fx in FIXTURES:
        state = run(fx["scenario"], thread_id=f"test-{fx['scenario']}", app=app)
        summary = build_summary(state)
        passed, fails = _check(fx, state, summary)
        results.append((fx["name"], passed, fails))

        if summary.get("status") == "ok":
            pipeline_ok += 1

        raised = len(state.get("anomalies", [])) > 0
        if fx["should_raise"] and raised:
            tp += 1
        elif fx["should_raise"] and not raised:
            fn += 1
        elif not fx["should_raise"] and raised:
            fp += 1
        else:
            tn += 1

        lc_total += 1
        if (summary.get("low_confidence_rca", 0) > 0) == fx["low_confidence"]:
            lc_correct += 1

        status = "PASS" if passed else "FAIL"
        print(f"\n[{status}] {fx['name']}")
        print(f"        anomalies={summary.get('anomalies_found')} "
              f"severity={summary.get('severity_breakdown')} "
              f"route={_actual_route(summary)} "
              f"low_conf_rca={summary.get('low_confidence_rca')}")
        for f in fails:
            print(f"        - {f}")

    n = len(FIXTURES)
    passed_n = sum(1 for _, p, _ in results if p)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print("\n" + "=" * 72)
    print("ACCURACY REPORT")
    print("=" * 72)
    print(f"Scenarios passed         : {passed_n}/{n}")
    print(f"Pipeline success rate    : {pipeline_ok}/{n} ({pipeline_ok / n:.0%})")
    print(f"Anomaly confusion matrix : TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Anomaly precision        : {precision:.2f}")
    print(f"Anomaly recall           : {recall:.2f}")
    print(f"False-positive rate      : {fpr:.2f}")
    print(f"Low-confidence handling  : {lc_correct}/{lc_total} correct "
          f"(admits uncertainty when retrieval is weak, stays confident otherwise)")
    print("=" * 72)

    all_passed = passed_n == n
    print(f"\nRESULT: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
