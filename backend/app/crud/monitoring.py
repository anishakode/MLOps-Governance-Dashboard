from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple, Dict, Any
import os, json, requests
from prometheus_client import Counter, Gauge
from sqlalchemy.orm import Session

from app.models.monitoring import MonitoringMetric, Alert
from app.models.audit import AuditLog
from app.utils.drift import population_stability_index, classify_psi
from app.utils.runtime_config import get_slack_webhook
from app.services.audit import record_audit
from app.utils.stats import ks_d_stat, ks_severity

APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000").rstrip("/")
COOLDOWN_MIN = int(os.getenv("ALERT_COOLDOWN_MIN", "30"))
DQ_CHECKS = Counter("dq_checks_total", "Number of data-quality checks", ["kind", "severity"])
DQ_ALERTS = Counter("dq_alerts_total", "Number of data-quality alerts", ["kind", "severity"])
SLACK_MENTION = os.getenv("SLACK_MENTION", "").strip().lower()
OPEN_ALERTS_BY_SEV = Gauge(
    "open_alerts_by_severity",
    "Number of open alerts by severity",
    ["severity"]
)

def _audit(db: Session, action: str, model_id: int, actor: Optional[str], details: dict):
    """Write an immutable audit record."""
    db.add(AuditLog(action=action, model_id=model_id, actor=actor, details=details))
    db.commit()

def _slack_send(payload: dict) -> None:
    """Slack POST; never crash the API. Resolves webhook dynamically each call."""
    url = (get_slack_webhook() or os.getenv("SLACK_WEBHOOK_URL", "")).strip()
    if not url:
        print("[SLACK] webhook not set; skipping send")
        return
    # Ensure a text fallback to avoid Slack 400s on some workspaces
    if "text" not in payload:
        payload = {**payload, "text": payload.get("fallback", "MLOps Governance notification")}
    try:
        # requests will set the JSON header for us
        r = requests.post(url, json=payload, timeout=10)
        print(f"[SLACK] POST status={r.status_code}")
        if r.status_code >= 300:
            print(f"[SLACK] error: {r.text[:300]}")
    except Exception as e:
        print(f"[SLACK] send error: {e}")

def _refresh_open_alert_gauges(db: Session, model_id: Optional[int] = None):
    q = db.query(Alert).filter(Alert.status == "open")
    if model_id is not None:
        q = q.filter(Alert.model_id == model_id)
    counts = {"low": 0, "medium": 0, "high": 0}
    for a in q.all():
        counts[a.severity] = counts.get(a.severity, 0) + 1
    for sev, val in counts.items():
        OPEN_ALERTS_BY_SEV.labels(severity=sev).set(val)


def set_psi_baseline(db: Session, model_id: int, feature: str, expected_bins: List[float],
    psi_medium: float | None = None, psi_high: float | None = None) -> MonitoringMetric:
    details = {"expected": list(expected_bins)}
    if psi_medium is not None or psi_high is not None:
        details["thresholds"] = {"psi": {"medium": psi_medium, "high": psi_high}}
    m = MonitoringMetric(
        model_id=model_id,
        feature=feature,
        method="psi_baseline",
        value=None,
        details=details,
    )
    db.add(m); db.commit(); db.refresh(m)
    record_audit(db, "SET_BASELINE", model_id, None, {"feature": feature, "metric_id": m.id})
    return m

def eval_psi(db: Session, model_id: int, feature: str, actual_bins: List[float]) -> Tuple[MonitoringMetric, Optional[Alert]]:
    """
    Compute PSI vs the most recent baseline for (model_id, feature).
    Creates a MonitoringMetric 'psi_eval'. If severity is medium/high, also creates an Alert.
    """
    baseline = (
        db.query(MonitoringMetric)
        .filter(
            MonitoringMetric.model_id == model_id,
            MonitoringMetric.feature == feature,
            MonitoringMetric.method == "psi_baseline",
        )
        .order_by(MonitoringMetric.created_at.desc())
        .first()
    )
    if not baseline:
        raise ValueError(f"No baseline found for feature '{feature}' and model_id={model_id}")

    expected_bins = list(baseline.details.get("expected", []))
    if not expected_bins:
        raise ValueError(f"Baseline for feature '{feature}' is missing 'expected' bins")

    psi = population_stability_index(expected_bins, actual_bins)

    # Prefer per-feature thresholds if present
    th = ((baseline.details or {}).get("thresholds") or {}).get("psi") or {}
    pm = th.get("medium")
    ph = th.get("high")
    if pm is not None and ph is not None:
        if psi >= ph:
            severity, msg = "high", "Major drift detected"
        elif psi >= pm:
            severity, msg = "medium", "Moderate drift detected"
        else:
            severity, msg = "low", "Drift within acceptable bounds"
    else:
        # fallback to library defaults
        severity, msg = classify_psi(psi)

    mm = MonitoringMetric(
        model_id=model_id,
        feature=feature,
        method="psi_eval",
        value=f"{psi:.6f}",
        details={"expected": baseline.details["expected"], "actual": actual_bins, "severity": severity},
    )
    db.add(mm)
    db.commit()
    db.refresh(mm)

    record_audit(db, "DRIFT_EVALUATED", model_id, None,
             {"feature": feature, "psi": psi, "severity": severity, "metric_id": mm.id})

    alert_obj: Optional[Alert] = None   
    if severity in ("medium", "high"):
        since = datetime.utcnow() - timedelta(minutes=COOLDOWN_MIN)

        recent_open = (
            db.query(Alert)
            .filter(
                Alert.model_id == model_id,
                Alert.type == "drift",
                Alert.severity == severity,
                Alert.status == "open",
                Alert.created_at >= since,
            )
            .order_by(Alert.created_at.desc())
            .all()
        )

        # Compare feature stored in Alert.details
        duplicate = any(((a.details or {}).get("feature") == feature) for a in recent_open)

        if not duplicate:
            alert_obj = Alert(
                model_id=model_id,
                type="drift",
                severity=severity,
                message=f"{msg} on feature '{feature}' (PSI={psi:.3f})",
                details={"psi": psi, "feature": feature, "metric_id": mm.id},
            )
            db.add(alert_obj)
            db.commit()
            db.refresh(alert_obj)
            _refresh_open_alert_gauges(db, model_id)

            # Audit alert creation
            record_audit(
                db,
                "ALERT_CREATED",
                model_id,
                None,
                {"alert_id": alert_obj.id, "severity": severity, "feature": feature},
            )

            # Best-effort Slack (won't break API on failure)
            try:
                created_iso = alert_obj.created_at.isoformat() if alert_obj.created_at else ""
                sev_emoji = {"high": "🚨", "medium": "⚠️"}.get(severity, "⚠️")
                payload = {
                    "text": f"{sev_emoji} Drift {severity.upper()} • model {model_id} • {feature} PSI={psi:.3f}",  # ← add this
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": f"{sev_emoji} Drift Alert — {severity.upper()}",
                                "emoji": True,
                            },
                        },
                        {
                            "type": "section",
                            "fields": [
                                {"type": "mrkdwn", "text": f"*Model ID:*\n{model_id}"},
                                {"type": "mrkdwn", "text": f"*Feature:*\n`{feature}`"},
                                {"type": "mrkdwn", "text": f"*PSI:*\n{psi:.3f}"},
                                {"type": "mrkdwn", "text": f"*Status:*\n{alert_obj.status}"},
                            ],
                        },
                        {
                            "type": "context",
                            "elements": [
                                {"type": "mrkdwn", "text": f"*Created:* {created_iso}"},
                                {"type": "mrkdwn", "text": f"*Metric ID:* {mm.id} • *Alert ID:* {alert_obj.id}"},
                            ],
                        },
                    ],
                }
                _slack_send(payload)
            except Exception:
                pass

        else:
            # Audit suppression (duplicate within cooldown)
            record_audit(
                db,
                "ALERT_SUPPRESSED",
                model_id,
                None,
                {
                    "reason": "duplicate_within_cooldown",
                    "cooldown_min": COOLDOWN_MIN,
                    "feature": feature,
                    "severity": severity,
                },
            )

    return mm, alert_obj

def list_alerts(db: Session, model_id: Optional[int] = None, status: Optional[str] = None):
    q = db.query(Alert)
    if model_id is not None:
        q = q.filter(Alert.model_id == model_id)
    if status is not None:
        q = q.filter(Alert.status == status)
    return q.order_by(Alert.created_at.desc()).all()

def resolve_alert(db: Session, alert_id: int, actor: Optional[str] = None):
    alert = db.get(Alert, alert_id)
    if not alert:
        return None
    alert.status = "resolved"
    alert.resolved_at = datetime.utcnow()
    db.commit(); db.refresh(alert)
    _refresh_open_alert_gauges(db, alert.model_id)
    record_audit(db, "ALERT_RESOLVED", alert.model_id, actor, {"alert_id": alert.id})
    return alert

def eval_ks(db: Session, model_id: int, feature: str, baseline_sample: list[float], current_sample: list[float]):
    d = ks_d_stat(baseline_sample, current_sample)
    sev = ks_severity(d)
    mm = MonitoringMetric(model_id=model_id, feature=feature, method="ks_eval",
                          value=f"{d:.6f}", details={"severity": sev})
    db.add(mm); db.commit(); db.refresh(mm)
    # (optional) create alert similar to PSI thresholds
    return {"d_stat": d, "severity": sev, "metric_id": mm.id}

def set_dq_thresholds(
    db: Session,
    model_id: int,
    feature: str,
    missing_medium: float | None = None,
    missing_high: float | None = None,
    range_medium: float | None = None,
    range_high: float | None = None,
) -> MonitoringMetric:
    details: Dict[str, Any] = {}
    if missing_medium is not None or missing_high is not None:
        details["missing"] = {"medium": missing_medium, "high": missing_high}
    if range_medium is not None or range_high is not None:
        details["range"] = {"medium": range_medium, "high": range_high}
    m = MonitoringMetric(
        model_id=model_id, feature=feature, method="dq_config",
        value=None, details=details
    )
    db.add(m); db.commit(); db.refresh(m)
    record_audit(db, "DQ_CONFIG_SET", model_id, None, {"feature": feature, **details})
    return m

# --- DQ: Missing percentage ---------------------------------------------------
def eval_dq_missing(
    db: Session,
    model_id: int,
    feature: str,
    total_count: int,
    missing_count: int,
    medium_thresh: float = 0.10,   
    high_thresh: float = 0.20      
):
    if total_count <= 0:
        raise ValueError("total_count must be > 0")
    if missing_count < 0 or missing_count > total_count:
        raise ValueError("missing_count must be between 0 and total_count")
    
    cfg = (
        db.query(MonitoringMetric)
        .filter(
            MonitoringMetric.model_id == model_id,
            MonitoringMetric.feature == feature,
            MonitoringMetric.method == "dq_config",
        )
        .order_by(MonitoringMetric.created_at.desc())
        .first()
    )
    if cfg and (cfg.details or {}).get("missing"):
        if cfg.details["missing"].get("medium") is not None:
            medium_thresh = cfg.details["missing"]["medium"]
        if cfg.details["missing"].get("high") is not None:
            high_thresh = cfg.details["missing"]["high"]

    rate = missing_count / total_count
    severity = "low"
    if rate >= high_thresh:
        severity = "high"
    elif rate >= medium_thresh:
        severity = "medium"

    mm = MonitoringMetric(
        model_id=model_id,
        feature=feature,
        method="dq_missing",
        value=f"{rate:.6f}",
        details={
            "severity": severity,
            "missing_rate": rate,
            "total_count": total_count,
            "missing_count": missing_count,
            "thresholds": {"medium": medium_thresh, "high": high_thresh},
        },
    )
    db.add(mm); db.commit(); db.refresh(mm)
    DQ_CHECKS.labels(kind="missing", severity=severity).inc()
    record_audit(db, "DQ_MISSING_EVALUATED", model_id, None,
                 {"feature": feature, "missing_rate": rate, "severity": severity, "metric_id": mm.id})
    DQ_CHECKS.labels(kind="missing", severity=severity).inc()

    alert_obj = None
    if severity in ("medium", "high"):
        since = datetime.utcnow() - timedelta(minutes=int(os.getenv("ALERT_COOLDOWN_MIN", "30")))
        recent_open = (
            db.query(Alert)
            .filter(
                Alert.model_id == model_id,
                Alert.type == "dq",
                Alert.severity == severity,
                Alert.status == "open",
                Alert.created_at >= since,
            ).order_by(Alert.created_at.desc())
            .all()
        )
        duplicate = any(((a.details or {}).get("feature") == feature and (a.details or {}).get("kind") == "missing")
                        for a in recent_open)
        if not duplicate:
            msg = f"High missing rate on '{feature}' ({rate:.1%})" if severity == "high" else f"Elevated missing rate on '{feature}' ({rate:.1%})"
            alert_obj = Alert(
                model_id=model_id,
                type="dq",
                severity=severity,
                message=msg,
                details={"kind": "missing", "missing_rate": rate, "feature": feature, "metric_id": mm.id},
            )
            db.add(alert_obj); db.commit(); db.refresh(alert_obj)
            _refresh_open_alert_gauges(db, model_id)
            record_audit(db, "ALERT_CREATED", model_id, None,
                         {"alert_id": alert_obj.id, "severity": severity, "feature": feature, "type": "dq"})
            DQ_ALERTS.labels(kind="missing", severity=severity).inc()
            
            # Slack notify
            try:
                sev_emoji = {"high": "🚨", "medium": "⚠️"}.get(severity, "ℹ️")
                payload = {
                    "text": f"{sev_emoji} DQ {severity.upper()} • model {model_id} • {feature} missing={rate:.1%}",
                    "blocks": [
                        {"type": "header", "text": {"type": "plain_text",
                         "text": f"{sev_emoji} Data Quality — {severity.upper()} (Missing %)", "emoji": True}},
                        {"type": "section", "fields": [
                            {"type": "mrkdwn", "text": f"*Model ID:*\n{model_id}"},
                            {"type": "mrkdwn", "text": f"*Feature:*\n`{feature}`"},
                            {"type": "mrkdwn", "text": f"*Missing:*\n{rate:.1%}"},
                            {"type": "mrkdwn", "text": f"*Status:*\n{alert_obj.status}"},
                        ]},
                        {"type": "context", "elements": [
                            {"type": "mrkdwn", "text": f"*Metric ID:* {mm.id} • *Alert ID:* {alert_obj.id}"},
                        ]},
                    ],
                }
                _slack_send(payload)
            except Exception:
                pass
        else:
            record_audit(db, "ALERT_SUPPRESSED", model_id, None,
                         {"reason": "duplicate_within_cooldown", "cooldown_min": int(os.getenv("ALERT_COOLDOWN_MIN", "30")),
                          "feature": feature, "severity": severity, "type": "dq", "kind": "missing"})
    return {"missing_rate": rate, "severity": severity, "metric_id": mm.id, "alert_id": getattr(alert_obj, "id", None)}


# --- DQ: Out-of-range fraction ------------------------------------------------
def eval_dq_range(
    db: Session,
    model_id: int,
    feature: str,
    total_count: int,
    out_of_range_count: int,
    expected_min: Optional[float] = None,
    expected_max: Optional[float] = None,
    observed_min: Optional[float] = None,
    observed_max: Optional[float] = None,
    medium_thresh: float = 0.10,   # 10%
    high_thresh: float = 0.20      # 20%
):
    if total_count <= 0:
        raise ValueError("total_count must be > 0")
    if out_of_range_count < 0 or out_of_range_count > total_count:
        raise ValueError("out_of_range_count must be between 0 and total_count")

    cfg = (
        db.query(MonitoringMetric)
        .filter(
            MonitoringMetric.model_id == model_id,
            MonitoringMetric.feature == feature,
            MonitoringMetric.method == "dq_config",
        )
        .order_by(MonitoringMetric.created_at.desc())
        .first()
    )
    if cfg and (cfg.details or {}).get("range"):
        if cfg.details["range"].get("medium") is not None:
            medium_thresh = cfg.details["range"]["medium"]
        if cfg.details["range"].get("high") is not None:
            high_thresh = cfg.details["range"]["high"]

    frac = out_of_range_count / total_count
    severity = "low"
    if frac >= high_thresh:
        severity = "high"
    elif frac >= medium_thresh:
        severity = "medium"

    details = {
        "severity": severity,
        "out_of_range_frac": frac,
        "total_count": total_count,
        "out_of_range_count": out_of_range_count,
        "thresholds": {"medium": medium_thresh, "high": high_thresh},
    }
    if expected_min is not None: details["expected_min"] = expected_min
    if expected_max is not None: details["expected_max"] = expected_max
    if observed_min is not None: details["observed_min"] = observed_min
    if observed_max is not None: details["observed_max"] = observed_max

    mm = MonitoringMetric(model_id=model_id, feature=feature, method="dq_range",
                          value=f"{frac:.6f}", details=details)
    db.add(mm); db.commit(); db.refresh(mm)
    _refresh_open_alert_gauges(db, model_id)
    record_audit(db, "DQ_RANGE_EVALUATED", model_id, None,
                 {"feature": feature, "out_of_range_frac": frac, "severity": severity, "metric_id": mm.id})
    DQ_CHECKS.labels(kind="range", severity=severity).inc()

    alert_obj = None
    if severity in ("medium", "high"):
        since = datetime.utcnow() - timedelta(minutes=int(os.getenv("ALERT_COOLDOWN_MIN", "30")))
        recent_open = (
            db.query(Alert)
            .filter(
                Alert.model_id == model_id,
                Alert.type == "dq",
                Alert.severity == severity,
                Alert.status == "open",
                Alert.created_at >= since,
            ).order_by(Alert.created_at.desc())
            .all()
        )
        duplicate = any(((a.details or {}).get("feature") == feature and (a.details or {}).get("kind") == "range")
                        for a in recent_open)
        if not duplicate:
            msg = f"High out-of-range fraction on '{feature}' ({frac:.1%})" if severity == "high" else f"Elevated out-of-range on '{feature}' ({frac:.1%})"
            alert_obj = Alert(
                model_id=model_id,
                type="dq",
                severity=severity,
                message=msg,
                details={"kind": "range", "out_of_range_frac": frac, "feature": feature, "metric_id": mm.id},
            )
            db.add(alert_obj); db.commit(); db.refresh(alert_obj)
            record_audit(db, "ALERT_CREATED", model_id, None,
                         {"alert_id": alert_obj.id, "severity": severity, "feature": feature, "type": "dq"})
            DQ_ALERTS.labels(kind="range", severity=severity).inc()
            
            try:
                sev_emoji = {"high": "🚨", "medium": "⚠️"}.get(severity, "ℹ️")
                payload = {
                    "text": f"{sev_emoji} DQ {severity.upper()} • model {model_id} • {feature} out_of_range={frac:.1%}",
                    "blocks": [
                        {"type": "header", "text": {"type": "plain_text",
                         "text": f"{sev_emoji} Data Quality — {severity.upper()} (Range)", "emoji": True}},
                        {"type": "section", "fields": [
                            {"type": "mrkdwn", "text": f"*Model ID:*\n{model_id}"},
                            {"type": "mrkdwn", "text": f"*Feature:*\n`{feature}`"},
                            {"type": "mrkdwn", "text": f"*Out-of-range:*\n{frac:.1%}"},
                            {"type": "mrkdwn", "text": f"*Status:*\n{alert_obj.status}"},
                        ]},
                        {"type": "context", "elements": [
                            {"type": "mrkdwn", "text": f"*Metric ID:* {mm.id} • *Alert ID:* {alert_obj.id}"},
                        ]},
                    ],
                }
                # include expected/observed if provided
                if expected_min is not None or expected_max is not None or observed_min is not None or observed_max is not None:
                    payload["blocks"].append({
                        "type": "context",
                        "elements": [{"type": "mrkdwn",
                                      "text": f"*Expected:* [{expected_min if expected_min is not None else '-'}, {expected_max if expected_max is not None else '-'}] • "
                                              f"*Observed:* [{observed_min if observed_min is not None else '-'}, {observed_max if observed_max is not None else '-'}]"}]
                    })
                _slack_send(payload)
            except Exception:
                pass
        else:
            record_audit(db, "ALERT_SUPPRESSED", model_id, None,
                         {"reason": "duplicate_within_cooldown", "cooldown_min": int(os.getenv("ALERT_COOLDOWN_MIN", "30")),
                          "feature": feature, "severity": severity, "type": "dq", "kind": "range"})
    return {"out_of_range_frac": frac, "severity": severity, "metric_id": mm.id, "alert_id": getattr(alert_obj, "id", None)}
