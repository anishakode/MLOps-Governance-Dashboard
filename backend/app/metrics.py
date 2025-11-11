# backend/app/metrics.py
from prometheus_client import Counter, Gauge, Histogram

# === Core metrics (definitions ONLY here) ===
psi_evals_total = Counter(
    "psi_evaluations_total", "Number of PSI evaluations"
)

alerts_created_total = Counter(
    "alerts_created_total", "Alerts created", ["type", "severity"]
)

dq_checks_total = Counter(
    "dq_checks_total", "Number of data-quality checks", ["kind", "severity"]
)

dq_alerts_total = Counter(
    "dq_alerts_total", "Number of data-quality alerts", ["kind", "severity"]
)

slo_checks_total = Counter(
    "slo_checks_total", "Number of SLO checks"
)

slo_alerts_total = Counter(
    "slo_alerts_total", "Number of SLO alerts", ["severity"]
)

request_latency_seconds = Histogram(
    "request_latency_seconds", "Request latency"
)

policy_blocks_total = Counter(
    "policy_blocks_total",   "Policy blocks"
)

open_alerts_gauge   = Gauge(
    "open_alerts_by_severity","Number of open alerts by severity", ["severity"]
)

def init_metrics_zero():
    # create label combos at 0 so Grafana never sees “no data”
    severities = ["low","medium","high"]
    kinds = ["missing","range"]
    types = ["drift","dq","slo"]

    for s in severities:
        open_alerts_gauge.labels(severity=s).set(0)
        slo_alerts_total.labels(severity=s).inc(0)
        for t in types:
            alerts_created_total.labels(type=t, severity=s).inc(0)
        for k in kinds:
            dq_checks_total.labels(kind=k, severity=s).inc(0)
            dq_alerts_total.labels(kind=k, severity=s).inc(0)

    # unlabeled counters – make them visible
    psi_evals_total.inc(0)
    slo_checks_total.inc(0)

# Small helper so everyone updates the same gauge in the same way
def refresh_open_alerts_gauge(db, model_id: int | None = None):
    from app.models.monitoring import Alert
    q = db.query(Alert).filter(Alert.status == "open")
    if model_id is not None:
        q = q.filter(Alert.model_id == model_id)

    counts = {"low": 0, "medium": 0, "high": 0}
    for a in q.all():
        counts[a.severity] = counts.get(a.severity, 0) + 1

    for sev in ("low", "medium", "high"):
        open_alerts_gauge.labels(severity=sev).set(counts.get(sev, 0))



