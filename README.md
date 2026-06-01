# OpsCortex

A multi-agent **operations-intelligence** system for SaaS / e-commerce / fintech
ops teams. It ingests operational signals, detects anomalies with deterministic
math, performs RAG-grounded root-cause analysis, drafts confidence-scored action
plans, routes risky actions through human approval (HITL), and writes an
append-only audit trail. Orchestrated as a LangGraph state machine.

It ships with a **deterministic offline LLM backend as the default** so the graph
and tests run green with no API key; the real Anthropic path is one env toggle
away. This is a reference implementation, not a turnkey production deployment —
see [Where it can fail](#where-it-can-fail).

---

## What it does

1. **Monitor** ops signals from four mock data sources (payments, support,
   app metrics, infra).
2. **Detect** anomalies — spikes, drops, threshold breaches, trend reversals —
   with deterministic detectors. The LLM never does the raw math.
3. **Root-cause analysis** via retrieval over a historical-incident store, with
   citations and a confidence score. If retrieval is weak it *admits
   uncertainty* instead of inventing a cause.
4. **Draft** a prioritized, confidence-scored action plan per incident.
5. **Route** high-risk / destructive / customer-facing / high-severity /
   low-confidence actions to **human approval (HITL)**; only safe, confident,
   low-risk actions auto-run.
6. **Log** every step to an append-only audit trail (in-memory + JSONL file).

## Architecture

```
ingest -> detect -> (anomalies?) --no--> END
                       | yes
                       v
                    triage -> rca -> plan -> route
                                               |
                                               v
                                        auto_action --(pending HITL plans?)--> hitl --> END
   any node raises  ------------------------------------------------> error -> END
```

| Node | Responsibility | Model |
|------|----------------|-------|
| `ingest` | Pull signals from the data-source adapters | — |
| `detect` | **Deterministic** anomaly detection → typed `Anomaly` objects | — |
| `triage` | Classify severity + priority | Haiku (`claude-haiku-4-5-20251001`) |
| `rca` | RAG over historical incidents; grounded hypothesis + citations + confidence | Sonnet (`claude-sonnet-4-6`) |
| `plan` | Draft ordered action steps with risk + confidence | Sonnet |
| `route` | Decide auto-action vs HITL from severity × confidence × risk | — |
| `auto_action` | Execute low-risk, high-confidence plans; audit | — |
| `hitl` | Request human approval via a non-TTY provider | — |
| `error` | Graceful failure handler | — |

Conditional edges; a `MemorySaver` checkpointer keys every run by `thread_id`.
Models are configurable constants in [`config.py`](config.py) (`SYNTHESIS_MODEL`,
`TRIAGE_MODEL`).

## The offline backend (default)

The LLM sits behind an interface with a **deterministic offline backend** that is
the default, so the graph and tests run green, free, and reproducibly with no key.
It produces schema-valid structured output from the deterministic detector facts —
it validates wiring/governance, **not** model-quality reasoning.

To use the real Anthropic models:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:OPSCORTEX_LLM_BACKEND = "anthropic"
```
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPSCORTEX_LLM_BACKEND=anthropic
```

If `anthropic` is requested without a key, it logs a warning and falls back to
offline rather than crashing.

## Setup

Python 3.10+ (developed on 3.14).

```bash
python -m pip install -r requirements.txt
```

## How to run

```bash
python main.py                          # default scenario (payment spike)
python main.py ticket_surge
python main.py latency_creep
python main.py noisy_benign
python main.py low_retrieval_confidence

python test_runner.py                   # test suite + accuracy report
```

Optional chat UI (real Anthropic backend): `python server.py` then open
http://127.0.0.1:8000. Files: [`server.py`](server.py), [`web_agent.py`](web_agent.py),
[`web/index.html`](web/index.html). The dev server has no auth — don't expose it
publicly as-is.

### Test results (offline, deterministic)

```
Scenarios passed         : 5/5
Pipeline success rate    : 5/5 (100%)
Anomaly precision        : 1.00
Anomaly recall           : 1.00
False-positive rate      : 0.00
Low-confidence handling  : 5/5 correct
```

These measure detector + routing + governance correctness on labeled fixtures —
not the analytical quality of a live LLM.

## Swapping mocks for real APIs

| Replace | In | With |
|---------|----|------|
| `PaymentEventsAdapter.fetch()` | [`adapters.py`](adapters.py) | Stripe API |
| `SupportTicketsAdapter.fetch()` | [`adapters.py`](adapters.py) | Zendesk API |
| `AppMetricsAdapter.fetch()` | [`adapters.py`](adapters.py) | Datadog / Prometheus |
| `InfraAlertsAdapter.fetch()` | [`adapters.py`](adapters.py) | Status page / Pingdom |
| `IncidentStore` / `retrieve` | [`rag.py`](rag.py) | Real embeddings + vector DB |
| `QueueApprovalProvider` | [`tools.py`](tools.py) | Slack / email / dashboard provider |

## Governance & safety

- Confidence scoring on every RCA and action plan.
- Destructive / customer-facing / high-severity / low-confidence plans always go
  to **HITL**, never auto-execute. Cutoffs in [`config.py`](config.py).
- HITL is **non-TTY**: the default `QueueApprovalProvider` returns `PENDING`
  (queued for an external approver); it never blocks on a terminal and never
  silently auto-approves.
- Append-only audit log: step, status, inputs, outputs, timestamp, model used —
  in state and as JSONL at `audit_log.jsonl`.

## Reliability

- Retry with exponential backoff on every live LLM call (max 3, `tenacity`).
- Robust JSON parsing: strip fences → regex-extract the outermost `{...}` → raise
  the real error on total failure (never fabricate).
- Token budgeting per node; Pydantic validation on every node output; any
  exception routes to the `error` node.

## Where it can fail

- The offline LLM is rule-based — it validates wiring, not analytical quality.
  The real backend carries normal LLM risks (errors, latency, cost).
- Lexical (TF-IDF) retrieval is weaker than semantic embeddings; it can miss
  paraphrased incidents or match on shared boilerplate. The low-confidence path
  mitigates but does not eliminate this.
- Detector thresholds are heuristic and dataset-dependent; expect tuning on real
  feeds.
- No persistence beyond the process (in-memory checkpointer; local audit JSONL).
- Action execution is mocked (`nodes._execute`) — it records intent, it does not
  call real runbooks. No claims of perfection or "zero hallucination."

## Files

| File | Purpose |
|------|---------|
| `config.py` | Models, backend toggle, thresholds, routing rules, budgets, logging |
| `state.py` | Pydantic models + the LangGraph `OpsState` schema |
| `adapters.py` | Typed mock data-source adapters (swap point for real APIs) |
| `detectors.py` | Deterministic detectors (threshold, z-score, %-change, trend reversal) |
| `rag.py` | Historical-incident corpus + TF-IDF cosine retrieval |
| `tools.py` | LLM client (offline/Anthropic), JSON parsing, budgeting, audit, HITL |
| `nodes.py` | LangGraph node functions + run summary |
| `graph.py` | Graph assembly, conditional edges, checkpointer |
| `main.py` | CLI runner |
| `test_runner.py` | Labeled-fixture suite + accuracy report |
| `server.py`, `web_agent.py`, `web/index.html` | Optional chat UI (real backend) |
