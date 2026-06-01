"""
rag.py — Historical-incident store + retrieval for root-cause analysis.

A small corpus of past incidents (cause + resolution) is embedded with a
pure-Python TF-IDF lexical vector and queried by cosine similarity. This keeps
RAG deterministic and dependency-free (no faiss/numpy, no API key).

Retrieval returns similarity scores so the RCA node can detect *low retrieval
confidence* and refuse to invent a cause.

PRODUCTION SWAP: replace `_embed` / `IncidentStore` internals with a real
embedding model (e.g. Anthropic/Voyage embeddings) + a vector DB. The
`retrieve()` return shape (incident dict + similarity) stays the same.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import config

# ---------------------------------------------------------------------------
# Historical incident corpus (18 past incidents)
# ---------------------------------------------------------------------------
HISTORICAL_INCIDENTS: list[dict] = [
    {"id": "INC-1001", "title": "Payment failure spike from processor outage",
     "text": "Payment failure rate spiked when the upstream card processor had a regional outage. "
             "Declined charges and gateway timeouts surged across checkout.",
     "root_cause": "Upstream payment processor regional outage causing gateway timeouts.",
     "resolution": "Failed over to the secondary processor and paused automatic retries."},
    {"id": "INC-1002", "title": "Payment declines from expired API key rotation",
     "text": "A botched payment API key rotation caused authentication failures and a spike in "
             "declined charges and payment failure rate.",
     "root_cause": "Expired/rotated payment API credentials rejected by the processor.",
     "resolution": "Restored the valid API key and re-validated the credential rotation runbook."},
    {"id": "INC-1003", "title": "Support ticket surge after broken release",
     "text": "Support ticket volume surged after a release broke the login flow; customers opened "
             "many tickets about being unable to sign in.",
     "root_cause": "Faulty release broke the authentication flow, driving a support ticket surge.",
     "resolution": "Rolled back the release; ticket volume returned to baseline within an hour."},
    {"id": "INC-1004", "title": "Support ticket surge from billing email bug",
     "text": "A billing email bug double-charged some customers, producing a support ticket surge "
             "and refund requests.",
     "root_cause": "Billing job ran twice and sent duplicate charge emails.",
     "resolution": "Disabled the duplicate job, issued refunds, and sent a correction email."},
    {"id": "INC-1005", "title": "API p95 latency increase from slow DB query",
     "text": "API p95 latency increased steadily after a new query without an index was deployed; "
             "latency crept up under normal load.",
     "root_cause": "Missing database index caused a slow query and rising p95 latency.",
     "resolution": "Added the index and the latency returned to baseline."},
    {"id": "INC-1006", "title": "Latency increase from connection pool exhaustion",
     "text": "API latency increased and requests queued when the database connection pool was "
             "exhausted during a traffic peak.",
     "root_cause": "Undersized DB connection pool exhausted under load.",
     "resolution": "Increased pool size and added pool saturation alerts."},
    {"id": "INC-1007", "title": "Error rate increase from bad deploy",
     "text": "Application error rate increased sharply right after a deploy that shipped a null "
             "pointer regression in the orders service.",
     "root_cause": "Regression in a deploy raised the application error rate.",
     "resolution": "Rolled back the deploy and added a regression test."},
    {"id": "INC-1008", "title": "Error rate increase from downstream dependency",
     "text": "Application error rate increased when a downstream inventory dependency returned 5xx "
             "errors and the service did not degrade gracefully.",
     "root_cause": "Downstream dependency 5xx errors propagated as application errors.",
     "resolution": "Added a circuit breaker and a cached fallback."},
    {"id": "INC-1009", "title": "Infrastructure uptime drop from zone outage",
     "text": "Infrastructure uptime dropped during a cloud availability-zone outage; health checks "
             "failed and nodes went unhealthy.",
     "root_cause": "Cloud availability-zone outage took a subset of nodes offline.",
     "resolution": "Shifted traffic to healthy zones and rebalanced capacity."},
    {"id": "INC-1010", "title": "Uptime drop from expired TLS certificate",
     "text": "Uptime dropped and outage alerts fired when a TLS certificate expired and clients "
             "could no longer connect.",
     "root_cause": "Expired TLS certificate broke client connections.",
     "resolution": "Renewed the certificate and automated renewal monitoring."},
    {"id": "INC-1011", "title": "Churn rate increase after pricing change",
     "text": "Customer churn rate increased in the weeks after an unannounced pricing change "
             "frustrated long-time customers.",
     "root_cause": "Pricing change without communication drove customer churn.",
     "resolution": "Grandfathered existing plans and improved change communication."},
    {"id": "INC-1012", "title": "Payment volume drop from checkout JS error",
     "text": "Payment volume dropped when a JavaScript error on the checkout page prevented order "
             "submission for some browsers.",
     "root_cause": "Checkout JavaScript regression blocked order submission.",
     "resolution": "Hotfixed the checkout script and added browser smoke tests."},
    {"id": "INC-1013", "title": "Error rate increase from rate-limit misconfiguration",
     "text": "Application error rate increased after a rate-limit misconfiguration rejected "
             "legitimate traffic with 429 errors.",
     "root_cause": "Overly aggressive rate-limit configuration rejected valid requests.",
     "resolution": "Tuned rate-limit thresholds and added per-tenant limits."},
    {"id": "INC-1014", "title": "Latency increase from cache stampede",
     "text": "API latency increased when a cache expired and a stampede of requests hit the "
             "database simultaneously.",
     "root_cause": "Cache expiry caused a stampede and database overload.",
     "resolution": "Added request coalescing and staggered cache TTLs."},
    {"id": "INC-1015", "title": "Payment failure rate spike from fraud-rule rollout",
     "text": "Payment failure rate spiked after a new fraud rule was too strict and blocked "
             "legitimate charges as suspected fraud.",
     "root_cause": "Overly strict fraud rule blocked legitimate payments.",
     "resolution": "Relaxed the fraud rule and added a shadow-mode evaluation step."},
    {"id": "INC-1016", "title": "Support ticket surge from outage spillover",
     "text": "Support ticket volume surged as a spillover effect of an infrastructure outage; "
             "customers reported the app being down.",
     "root_cause": "Infrastructure outage caused a support ticket surge.",
     "resolution": "Posted a status page update and a proactive customer notice."},
    {"id": "INC-1017", "title": "Uptime drop from failed deployment rollout",
     "text": "Uptime dropped during a deployment that crash-looped pods, failing health checks and "
             "reducing available capacity.",
     "root_cause": "Bad deployment crash-looped and failed health checks.",
     "resolution": "Rolled back and added readiness-probe gating to the rollout."},
    {"id": "INC-1018", "title": "Latency increase from noisy-neighbor batch job",
     "text": "API p95 latency increased when a heavy batch job saturated shared CPU, slowing "
             "online request handling.",
     "root_cause": "Batch job contended for CPU with online traffic.",
     "resolution": "Moved the batch job to an isolated worker pool and scheduled off-peak."},
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
         "was", "were", "is", "are", "by", "as", "that", "this", "from", "at",
         "when", "after", "some", "did", "not", "no"}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 2]


class IncidentStore:
    """Pure-Python TF-IDF + cosine vector store over the incident corpus."""

    def __init__(self, incidents: list[dict]):
        self.incidents = incidents
        self._docs_tokens = [
            _tokens(f"{i['title']} {i['text']} {i['root_cause']}") for i in incidents
        ]
        # IDF over the corpus.
        n = len(incidents)
        df: Counter[str] = Counter()
        for toks in self._docs_tokens:
            for t in set(toks):
                df[t] += 1
        self._idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
        self._doc_vectors = [self._vectorize(toks) for toks in self._docs_tokens]

    def _vectorize(self, toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        vec = {t: c * self._idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            vec = {t: v / norm for t, v in vec.items()}
        return vec

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(t, 0.0) for t, v in a.items())

    def retrieve(self, query: str, k: int = config.RAG_TOP_K) -> list[tuple[dict, float]]:
        """Return up to k (incident, similarity) pairs sorted by similarity."""
        qvec = self._vectorize(_tokens(query))
        if not qvec:
            return []
        scored = [(inc, self._cosine(qvec, dv))
                  for inc, dv in zip(self.incidents, self._doc_vectors)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# Module-level singleton store.
_STORE = IncidentStore(HISTORICAL_INCIDENTS)


def retrieve(query: str, k: int = config.RAG_TOP_K) -> list[tuple[dict, float]]:
    return _STORE.retrieve(query, k)
