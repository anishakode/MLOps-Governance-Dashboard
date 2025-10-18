from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from sqlalchemy.orm import Session

from app.models.model import Model
from app.models.approval import ApprovalRequest
from app.models.audit import AuditLog
from app.models.monitoring import MonitoringMetric, Alert
from app.utils.policy import get_policy

def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _serialize_approval(a: ApprovalRequest) -> Dict[str, Any]:
    return {
        "id": a.id,
        "model_id": a.model_id,
        "target": a.target,
        "status": a.status,
        "requested_by": a.requested_by,
        "justification": a.justification,
        "created_at": _iso(a.created_at),
        "decisions": a.decisions or [],
    }


def _serialize_audit(e: AuditLog) -> Dict[str, Any]:
    return {
        "id": e.id,
        "action": e.action,
        "model_id": e.model_id,
        "actor": e.actor,
        "details": e.details or {},
        "created_at": _iso(e.created_at),
    }


def _serialize_metric(m: MonitoringMetric) -> Dict[str, Any]:
    return {
        "id": m.id,
        "feature": m.feature,
        "method": m.method,
        "value": m.value,
        "details": m.details or {},
        "created_at": _iso(m.created_at),
    }


def _serialize_alert(a: Alert) -> Dict[str, Any]:
    return {
        "id": a.id,
        "type": a.type,
        "severity": a.severity,
        "message": a.message,
        "details": a.details or {},
        "status": a.status,
        "created_at": _iso(a.created_at),
        "resolved_at": _iso(a.resolved_at),
    }


def generate_audit_pack(
    db: Session,
    model_id: int,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Aggregate model + approvals + audit log + drift metrics + alerts into one JSON."""
    model = db.get(Model, model_id)
    pol = get_policy()
    if not model:
        raise ValueError(f"Model {model_id} not found")

    if end is None:
        end = datetime.utcnow()
    if start is None:
        start = end - timedelta(days=30)

    # Core
    model_info = {
        "id": model.id,
        "name": model.name,
        "description": model.description,
        "version": model.version,
        "stage": model.stage,
        "status": model.status,
        "created_at": _iso(model.created_at),
        "updated_at": _iso(getattr(model, "updated_at", None)),
        "model_metadata": getattr(model, "model_metadata", None),
    }

    approvals: List[ApprovalRequest] = (
        db.query(ApprovalRequest)
        .filter(
            ApprovalRequest.model_id == model_id,
            ApprovalRequest.created_at >= start,
            ApprovalRequest.created_at <= end,
        )
        .order_by(ApprovalRequest.created_at.desc())
        .all()
    )

    audits: List[AuditLog] = (
        db.query(AuditLog)
        .filter(
            AuditLog.model_id == model_id,
            AuditLog.created_at >= start,
            AuditLog.created_at <= end,
        )
        .order_by(AuditLog.created_at.desc())
        .all()
    )

    # Metrics
    baselines: List[MonitoringMetric] = (
        db.query(MonitoringMetric)
        .filter(
            MonitoringMetric.model_id == model_id,
            MonitoringMetric.method == "psi_baseline",
        )
        .order_by(MonitoringMetric.feature.asc(), MonitoringMetric.created_at.desc())
        .all()
    )

    evals: List[MonitoringMetric] = (
        db.query(MonitoringMetric)
        .filter(
            MonitoringMetric.model_id == model_id,
            MonitoringMetric.method == "psi_eval",
            MonitoringMetric.created_at >= start,
            MonitoringMetric.created_at <= end,
        )
        .order_by(MonitoringMetric.created_at.desc())
        .all()
    )

    alerts: List[Alert] = (
        db.query(Alert)
        .filter(
            Alert.model_id == model_id,
            Alert.created_at >= start,
            Alert.created_at <= end,
        )
        .order_by(Alert.created_at.desc())
        .all()
    )

    # Build pack
    pack: Dict[str, Any] = {
        "generated_at": _iso(datetime.utcnow()),
        "window": {"start": _iso(start), "end": _iso(end)},
        "model": model_info,
        "approvals": [_serialize_approval(a) for a in approvals],
        "audit_log": [_serialize_audit(a) for a in audits],
        "metrics": {
            "baselines": [_serialize_metric(m) for m in baselines],
            "psi_evaluations": [_serialize_metric(m) for m in evals],
        },
        "alerts": [_serialize_alert(a) for a in alerts],
        "summary": {
            "open_alerts": sum(1 for a in alerts if a.status == "open"),
            "high_alerts": sum(1 for a in alerts if a.severity == "high"),
            "evaluations_in_window": len(evals),
            "approvals_in_window": len(approvals),
            "policy_hash": pol.get("__hash"),
        },
    }

    return pack
