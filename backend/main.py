from fastapi import FastAPI, HTTPException, Depends, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy.orm import Session
from sqlalchemy import text, inspect, func
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal
from datetime import datetime, timezone, timedelta
from mlflow.tracking import MlflowClient
from collections import defaultdict
from pathlib import Path
import mlflow, uvicorn, asyncio, os, requests, json

# Import our modules
from app.core.database import get_db, engine, Base, SessionLocal
from app.models.model import Model as ModelDB, ModelStage, ModelStatus
from app.crud.model import model_crud
from app.crud.approval import request_promotion, add_decision, ApprovalRequest
from app.models.audit import AuditLog
from app.crud.monitoring import eval_slo, set_psi_baseline, eval_psi, list_alerts, resolve_alert,_slack_send, eval_ks, eval_dq_missing, eval_dq_range, set_dq_thresholds
from app.models.monitoring import MonitoringMetric, Alert
from app.utils.runtime_config import set_slack_webhook, get_slack_webhook
from app.utils.report import generate_audit_pack
from app.utils.audit_sink import write_event, AUDIT_DIR
from app.utils.pack_sink import write_pack, PACK_DIR
from app.utils.policy import get_policy, reload_policy, explain_policy
from app.crud.waiver import create_waiver, revoke_waiver, list_waivers
from app.models.waiver import Waiver
from app.services.audit import record_audit
from app.core.security import create_access_token, create_refresh_token, decode_token, ACCESS_TTL_MIN
from app.deps.auth import require_role, get_current_user

PSI_EVALS = Counter("psi_evaluations_total", "Number of PSI evaluations")
ALERTS_CREATED = Counter("alerts_created_total", "Alerts created", ["severity"])
POLICY_BLOCKS = Counter("policy_blocks_total", "Policy blocks")
REQ_LAT = Histogram("request_latency_seconds", "Request latency")

MONITOR_ENABLED = os.getenv("MONITOR_ENABLED", "1") == "1"
MONITOR_INTERVAL_SEC = int(os.getenv("MONITOR_INTERVAL_SEC", "60"))
MONITOR_ONLY_STAGE = os.getenv("MONITOR_ONLY_STAGE", "production")

_scheduler_task = None  # asyncio.Task

print("DATABASE CONFIGURATION:")
print(f"Engine URL: {engine.url}")
print(f"Database type: {engine.name}")

# FastAPI app
app = FastAPI(
    title="MLOps Governance Dashboard API",
    description="Production-ready MLOps governance API",
    version="0.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.on_event("startup")
def on_startup():
    print("Creating tables on startup...")
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully!")
    except Exception as e:
        print(f"Error creating tables: {e}")
    insp = inspect(engine)
    print("Tables now:", insp.get_table_names())
    print("Audit packs dir:", PACK_DIR)

    global _scheduler_task
    if MONITOR_ENABLED:
        print(f"[monitor] enabled; interval={MONITOR_INTERVAL_SEC}s stage_filter='{MONITOR_ONLY_STAGE}'")
        loop = asyncio.get_event_loop()
        _scheduler_task = loop.create_task(_monitor_loop())
    else:
        print("[monitor] disabled by MONITOR_ENABLED=0")

@app.on_event("shutdown")
def on_shutdown():
    global _scheduler_task
    if _scheduler_task:
        _scheduler_task.cancel()
        _scheduler_task = None

# Pydantic models
class ModelCreate(BaseModel):
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"

class ModelResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    version: str
    stage: str
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class AlertCommentIn(BaseModel):
    comment: str

@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    try:
        # Test database connection
        models_count = db.query(ModelDB).count()
        tables = inspect(engine).get_table_names()
        return {
            "status": "healthy",
            "database": "connected",
            "models_count": models_count,
            "tables": tables,
            "timestamp": datetime.now()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now()
        }

# Endpoints
@app.post("/api/models", response_model=ModelResponse)
def create_model(model: ModelCreate, db: Session = Depends(get_db)):
    # Check if model name exists
    existing = db.query(ModelDB).filter(ModelDB.name == model.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Model name already exists")
    
    db_model = model_crud.create_model(db, model.dict())
    return ModelResponse.model_validate(db_model)

@app.get("/api/models", response_model=List[ModelResponse])
def get_models(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    models = model_crud.get_models(db, skip=skip, limit=limit)
    return [ModelResponse.model_validate(model) for model in models]

@app.get("/api/models/{model_id}", response_model=ModelResponse)
def get_model(model_id: int, db: Session = Depends(get_db)):
    model = model_crud.get_model(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return ModelResponse.model_validate(model)

@app.post("/api/models/{model_id}/stage")
def update_model_stage(model_id: int, stage: str, db: Session = Depends(get_db)):
    try:
        model_stage = ModelStage(stage)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid stage")
    
    model = model_crud.update_model_stage(db, model_id, model_stage)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {"message": f"Model stage updated to {stage}", "model": ModelResponse.from_orm(model)}

class PromotionRequest(BaseModel):
    target: str                 # "staging" | "production"
    justification: Optional[str] = None
    requested_by: str

@app.post("/api/models/{model_id}/promotions", response_model=dict)
def create_promotion(model_id: int, payload: PromotionRequest, db: Session = Depends(get_db)):
    m = db.get(ModelDB, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    if payload.target not in ("staging", "production"):
        raise HTTPException(status_code=400, detail="Invalid target")
    req = request_promotion(db, model_id, payload.target, payload.justification or "", payload.requested_by)
    return {"approval_id": req.id, "status": req.status}

class Decision(BaseModel):
    decided_by: str
    decision: str               
    note: Optional[str] = None

@app.post("/api/models/{model_id}/alerts/resolve_all", response_model=dict)
def api_resolve_all_alerts(
    model_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_role("approver","admin")),
):
    rows = (
        db.query(Alert)
        .filter(Alert.model_id == model_id, Alert.status == "open")
        .all()
    )
    count = 0
    for a in rows:
        a.status = "resolved"
        a.resolved_at = datetime.utcnow()
        count += 1
    db.commit()
    record_audit(db, "ALERTS_RESOLVED_BULK", model_id, getattr(user, "username", None), {"count": count})
    return {"resolved": count}

class SLOThresholdsIn(BaseModel):
    thresholds: Dict[str, float]
    directions: Optional[Dict[str, str]] = None 
    severity: Optional[Dict[str, str]] = None       

class SLOCheckIn(BaseModel):
    metrics: Optional[Dict[str, float]] = None
    tracking_uri: Optional[str] = None
    run_id: Optional[str] = None

@app.get("/api/models/{model_id}/slo", response_model=dict)
def api_get_slo(model_id: int, db: Session = Depends(get_db),
                user=Depends(require_role("viewer","approver","admin"))):
    m = db.get(ModelDB, model_id)
    if not m:
        raise HTTPException(404, "Model not found")
    meta = dict(getattr(m, "model_metadata", {}) or {})
    slo = dict(meta.get("slo", {}) or {})
    return {
        "thresholds": slo.get("thresholds") or {},
        "directions": slo.get("directions") or {},
        "severity":   slo.get("severity")   or {},
    }

@app.delete("/api/models/{model_id}/slo", response_model=dict)
def api_clear_slo(model_id: int, db: Session = Depends(get_db),
                  user=Depends(require_role("admin"))):
    m = db.get(ModelDB, model_id)
    if not m:
        raise HTTPException(404, "Model not found")
    meta = dict(getattr(m, "model_metadata", {}) or {})
    if "slo" in meta:
        meta.pop("slo", None)
        m.model_metadata = meta
        db.commit(); db.refresh(m)
        record_audit(db, "SLO_THRESHOLDS_CLEARED", model_id, getattr(user,"username",None), {})
    return {"cleared": True}

@app.post("/api/models/{model_id}/slo/thresholds", response_model=dict)
def api_set_slo_thresholds(model_id: int, body: SLOThresholdsIn,
                           db: Session = Depends(get_db),
                           user=Depends(require_role("admin"))):
    m = db.get(ModelDB, model_id)
    if not m:
        raise HTTPException(404, "Model not found")
    meta = dict(getattr(m, "model_metadata", {}) or {})
    slo = dict(meta.get("slo", {}) or {})
    slo["thresholds"] = body.thresholds
    if body.directions: slo["directions"] = body.directions
    if body.severity:   slo["severity"]   = body.severity
    meta["slo"] = slo
    m.model_metadata = meta
    db.commit(); db.refresh(m)
    record_audit(db, "SLO_THRESHOLDS_SET", model_id, getattr(user, "username", None), slo)
    return {"saved": True, "threshold_keys": list(body.thresholds.keys())}

@app.post("/api/models/{model_id}/slo/check", response_model=dict)
def api_slo_check(model_id: int, body: SLOCheckIn,
                  db: Session = Depends(get_db),
                  user=Depends(require_role("approver","admin"))):
    m = db.get(ModelDB, model_id)
    if not m:
        raise HTTPException(404, "Model not found")

    # resolve thresholds from model metadata
    meta = dict(getattr(m, "model_metadata", {}) or {})
    slo = dict(meta.get("slo", {}) or {})
    thresholds = slo.get("thresholds") or {}
    if not thresholds:
        raise HTTPException(400, "No SLO thresholds configured for this model")

    directions = slo.get("directions") or {}
    per_sev    = slo.get("severity") or {}

    metrics: Dict[str, float] = {}
    if body.metrics:
        metrics = {k: float(v) for k, v in body.metrics.items()}
    elif body.tracking_uri and body.run_id:
        try:
            client = MlflowClient(tracking_uri=body.tracking_uri)
            run = client.get_run(body.run_id)
            for k, v in run.data.metrics.items():
                try: metrics[k] = float(v)
                except: pass
        except Exception as e:
            print(f"[SLO] MLflow fetch failed: {e}")
    if not metrics:
        mf = ((meta.get("mlflow") or {}).get("metrics") or {})
        metrics = {k: float(v) for k, v in mf.items() if isinstance(v, (int, float, str))}
        # coerce strings
        for k, v in list(metrics.items()):
            try: metrics[k] = float(v)
            except: metrics.pop(k, None)

    if not metrics:
        raise HTTPException(400, "No metrics available to evaluate SLO")

    mm, alert = eval_slo(db, model_id, metrics, thresholds, directions, per_sev)
    out = {"metric_id": mm.id, "violations": len(mm.details.get("violations") or [])}
    if alert: out["alert_id"] = alert.id; out["severity"] = alert.severity
    return out

@app.post("/api/approvals/{approval_id}/decision", response_model=dict)
def decide(approval_id: int, payload: Decision, db: Session = Depends(get_db), user=Depends(require_role("approver", "admin"))):
    if payload.decision not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="Invalid decision")
    req = add_decision(db, approval_id, payload.decided_by, payload.decision, payload.note)
    if not req:
        raise HTTPException(status_code=404, detail="Approval not found")
    return {"approval_id": req.id, "status": req.status, "decisions": req.decisions}

class AuditEntry(BaseModel):
    id: int
    action: str
    model_id: Optional[int]
    actor: Optional[str]
    details: dict
    created_at: datetime
    class Config:
        from_attributes = True

@app.get("/api/models/{model_id}/audit", response_model=List[AuditEntry])
def get_audit(model_id: int, limit: int = 50, db: Session = Depends(get_db)):
    rows = (
        db.query(AuditLog)
        .filter(AuditLog.model_id == model_id)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [AuditEntry.model_validate(r) for r in rows]

class ApprovalSummary(BaseModel):
    id: int
    model_id: int
    target: str
    status: str
    requested_by: Optional[str] = None
    created_at: datetime
    decisions: List[dict] = []
    class Config:
        from_attributes = True

@app.get("/api/approvals", response_model=List[ApprovalSummary])
def list_approvals(status: Optional[str] = None, db: Session = Depends(get_db)):
    q = db.query(ApprovalRequest)
    if status:
        q = q.filter(ApprovalRequest.status == status)
    rows = q.order_by(ApprovalRequest.created_at.desc()).all()
    return [ApprovalSummary.model_validate(r) for r in rows]

class MlflowSyncIn(BaseModel):
    model_id: int
    # Option A: send metrics/tags directly in the request body
    metrics: Optional[Dict[str, float]] = None
    tags: Optional[Dict[str, str]] = None
    # Option B: fetch from a live MLflow server (best effort)
    tracking_uri: Optional[str] = None
    run_id: Optional[str] = None

@app.post("/api/integrations/mlflow/sync")
def api_mlflow_sync(body: MlflowSyncIn, db: Session = Depends(get_db), user=Depends(require_role("approver", "admin"))):
    m = db.get(ModelDB, body.model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")

    # Start with whatever the caller provided
    mf_metrics: Dict[str, float] = dict(body.metrics or {})
    mf_tags: Dict[str, str] = dict(body.tags or {})

    # Pull from MLflow if tracking_uri + run_id are provided
    if body.tracking_uri and body.run_id:
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient(tracking_uri=body.tracking_uri)
            run = client.get_run(body.run_id)
            # latest metric values
            for k, v in run.data.metrics.items():
                try:
                    mf_metrics[k] = float(v)
                except Exception:
                    pass
            # tags as strings
            for k, v in run.data.tags.items():
                mf_tags[k] = str(v)
        except Exception as e:
            # Don't fail the request; just log the issue and proceed with whatever we have
            print(f"[MLflow] fetch failed: {e}")

    # Merge into model_metadata (JSON column)
    meta = dict(getattr(m, "model_metadata", {}) or {})
    mlflow_blob = dict(meta.get("mlflow", {}) or {})
    if mf_metrics:
        mlflow_blob["metrics"] = {**(mlflow_blob.get("metrics") or {}), **mf_metrics}
    if mf_tags:
        mlflow_blob["tags"] = {**(mlflow_blob.get("tags") or {}), **mf_tags}
    meta["mlflow"] = mlflow_blob
    m.model_metadata = meta

    db.commit()
    db.refresh(m)

    return {"status": "ok", "model_id": m.id, "model_metadata": m.model_metadata}

class BaselineIn(BaseModel):
    feature: str
    expected: List[float]
    psi_medium: Optional[float] = None   
    psi_high: Optional[float] = None     

@app.post("/api/models/{model_id}/drift/baseline", response_model=dict)
def api_set_baseline(model_id: int, payload: BaselineIn, db: Session = Depends(get_db),
                     user=Depends(require_role("approver","admin"))):
    m = db.get(ModelDB, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    rec = set_psi_baseline(
        db, model_id, payload.feature, payload.expected,
        psi_medium=payload.psi_medium, psi_high=payload.psi_high
    )
    return {"metric_id": rec.id, "message": f"Baseline set for {payload.feature}"}

class PsiEvalIn(BaseModel):
    feature: str
    actual: List[float]

@app.post("/api/models/{model_id}/drift/psi", response_model=dict)
def api_eval_psi(model_id: int, payload: PsiEvalIn, db: Session = Depends(get_db)):
    m = db.get(ModelDB, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        metric, alert = eval_psi(db, model_id, payload.feature, payload.actual)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    out = {"metric_id": metric.id, "psi": float(metric.value) if metric.value else None, "severity": metric.details.get("severity")}
    if alert:
        out["alert_id"] = alert.id
    return out

class PsiActualIn(BaseModel):
    feature: str
    actual: List[float] 

# ---- SNAPSHOT ENDPOINTS ----
@app.post("/api/models/{model_id}/drift/psi/actual", response_model=dict)
def api_push_psi_actual(
    model_id: int,
    body: PsiActualIn,
    db: Session = Depends(get_db),
    user=Depends(require_role("approver","admin")),
):
    m = db.get(ModelDB, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    if not body.actual or abs(sum(body.actual) - 1.0) > 1e-3:
        raise HTTPException(status_code=400, detail="actual bins must sum to ~1.0")
    row = MonitoringMetric(
        model_id=model_id,
        feature=body.feature,
        method="psi_actual",
        value=None,
        details={"actual": list(body.actual)},
    )
    db.add(row); db.commit(); db.refresh(row)
    record_audit(db, "PSI_SNAPSHOT", model_id, getattr(user, "username", None),
                 {"feature": body.feature, "metric_id": row.id})
    return {"snapshot_id": row.id}

class DQSnapshotIn(BaseModel):
    feature: str
    kind: Literal["missing", "range"]
    total_count: int
    # missing kind
    missing_count: Optional[int] = None
    # range kind
    out_of_range_count: Optional[int] = None
    expected_min: Optional[float] = None
    expected_max: Optional[float] = None
    observed_min: Optional[float] = None
    observed_max: Optional[float] = None

@app.post("/api/models/{model_id}/drift/dq/snapshot", response_model=dict)
def api_push_dq_snapshot(
    model_id: int,
    body: DQSnapshotIn,
    db: Session = Depends(get_db),
    user=Depends(require_role("approver","admin")),
):
    m = db.get(ModelDB, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    if body.total_count <= 0:
        raise HTTPException(status_code=400, detail="total_count must be > 0")
    if body.kind == "missing":
        if body.missing_count is None:
            raise HTTPException(400, "missing_count required for kind=missing")
        if body.missing_count < 0 or body.missing_count > body.total_count:
            raise HTTPException(400, "missing_count out of bounds")
        details = {"kind": "missing", "total_count": body.total_count, "missing_count": body.missing_count}
    else:  # range
        if body.out_of_range_count is None:
            raise HTTPException(400, "out_of_range_count required for kind=range")
        if body.out_of_range_count < 0 or body.out_of_range_count > body.total_count:
            raise HTTPException(400, "out_of_range_count out of bounds")
        details = {
            "kind": "range",
            "total_count": body.total_count,
            "out_of_range_count": body.out_of_range_count,
        }
        # optional extras
        if body.expected_min is not None: details["expected_min"] = body.expected_min
        if body.expected_max is not None: details["expected_max"] = body.expected_max
        if body.observed_min is not None: details["observed_min"] = body.observed_min
        if body.observed_max is not None: details["observed_max"] = body.observed_max

    row = MonitoringMetric(
        model_id=model_id,
        feature=body.feature,
        method="dq_snapshot",
        value=None,
        details=details,
    )
    db.add(row); db.commit(); db.refresh(row)
    record_audit(db, "DQ_SNAPSHOT", model_id, getattr(user, "username", None),
                 {"feature": body.feature, "metric_id": row.id, "kind": body.kind})
    return {"snapshot_id": row.id}

class AlertOut(BaseModel):
    id: int
    model_id: int
    type: str
    severity: str
    message: str
    details: dict
    status: str
    created_at: datetime
    resolved_at: Optional[datetime] = None
    class Config:
        from_attributes = True

@app.get("/api/alerts", response_model=List[AlertOut])
def api_list_alerts(model_id: Optional[int] = None, status: Optional[str] = None, db: Session = Depends(get_db)):
    q = db.query(Alert)
    if model_id is not None:
        q = q.filter(Alert.model_id == model_id)
    if status is not None:
        q = q.filter(Alert.status == status)
    rows = q.order_by(Alert.created_at.desc()).all()
    print(f"[alerts] returning {len(rows)} rows (model_id={model_id}, status={status})")
    return [AlertOut.model_validate(r) for r in rows]

@app.get("/debug/alerts/raw")
def debug_alerts_raw(model_id: Optional[int] = None, db: Session = Depends(get_db)):
    q = db.query(Alert)
    if model_id is not None:
        q = q.filter(Alert.model_id == model_id)
    rows = q.order_by(Alert.created_at.desc()).all()
    out = []
    for a in rows:
        out.append({
            "id": a.id,
            "model_id": a.model_id,
            "type": a.type,
            "severity": a.severity,
            "status": a.status,
            "message": a.message,
            "created_at": a.created_at.isoformat() if a.created_at else None,
            "details": a.details,
        })
    print(f"[debug/alerts/raw] rows={len(out)} for model_id={model_id}")
    return out


@app.post("/api/alerts/{alert_id}/resolve", response_model=AlertOut)
def api_resolve_alert(alert_id: int, db: Session = Depends(get_db)):
    r = resolve_alert(db, alert_id)
    if not r:
        raise HTTPException(status_code=404, detail="Alert not found")
    return AlertOut.model_validate(r)

class SlackWebhookIn(BaseModel):
    webhook_url: str

# Acknowledge (set status = acknowledged)
@app.post("/api/alerts/{alert_id}/ack", response_model=AlertOut)
def api_ack_alert(alert_id: int,
                  db: Session = Depends(get_db),
                  user=Depends(require_role("approver","admin"))):
    a = db.get(Alert, alert_id)
    if not a:
        raise HTTPException(404, "Alert not found")
    if a.status == "resolved":
        raise HTTPException(400, "Alert already resolved")
    a.status = "acknowledged"
    db.commit(); db.refresh(a)
    record_audit(db, "ALERT_ACKED", a.model_id, getattr(user, "username", None), {"alert_id": a.id})
    return AlertOut.model_validate(a)


# Comment on an alert (stored in audit trail)
@app.post("/api/alerts/{alert_id}/comment", response_model=dict)
def api_comment_alert(alert_id: int, body: AlertCommentIn,
                      db: Session = Depends(get_db),
                      user=Depends(require_role("approver","admin"))):
    a = db.get(Alert, alert_id)
    if not a:
        raise HTTPException(404, "Alert not found")
    record_audit(db, "ALERT_COMMENTED", a.model_id, getattr(user, "username", None),
                 {"alert_id": a.id, "comment": body.comment})
    return {"alert_id": a.id, "comment": body.comment, "recorded": True}


# Per-model alert summary (counts by status + severity)
@app.get("/api/models/{model_id}/alerts/summary", response_model=dict)
def api_alert_summary(model_id: int,
                      db: Session = Depends(get_db),
                      user=Depends(require_role("viewer","approver","admin"))):
    rows = db.query(Alert).filter(Alert.model_id == model_id).all()
    summary = {
        "total": len(rows),
        "by_status": {"open": 0, "acknowledged": 0, "resolved": 0},
        "by_severity": {"low": 0, "medium": 0, "high": 0},
    }
    for r in rows:
        summary["by_status"][r.status] = summary["by_status"].get(r.status, 0) + 1
        summary["by_severity"][r.severity] = summary["by_severity"].get(r.severity, 0) + 1
    return summary


@app.post("/config/slack-webhook", response_model=dict)
def api_set_slack_webhook(body: SlackWebhookIn, user=Depends(require_role("admin"))):
    url = body.webhook_url.strip()
    if not url.startswith("https://hooks.slack.com/"):
        raise HTTPException(status_code=400, detail="Invalid Slack webhook URL")
    set_slack_webhook(url)
    return {"saved": True}

@app.get("/config/slack-webhook", response_model=dict)
def api_get_slack_webhook(user=Depends(require_role("viewer", "approver", "admin"))):
    val = get_slack_webhook()
    masked = (val[:20] + "…") if val else None
    return {"configured": bool(val), "webhook_url_preview": masked}

@app.post("/debug/slack", response_model=dict)
def api_debug_slack():
    ok = _slack_send({"text": "✅ Debug: app can reach Slack (runtime-config)"})
    return {"sent": bool(ok), "configured": bool(get_slack_webhook())}

@app.post("/debug/slack/verbose", response_model=dict)
def api_debug_slack_verbose():
    url = (get_slack_webhook() or os.getenv("SLACK_WEBHOOK_URL","")).strip()
    if not url:
        return {"configured": False, "sent": False, "reason": "no_webhook"}
    try:
        r = requests.post(url, data=json.dumps({"text": "🔎 verbose Slack test"}), headers={"Content-Type": "application/json"}, timeout=6)
        return {"configured": True, "sent": r.status_code == 200, "status": r.status_code, "response": r.text[:300]}
    except Exception as e:
        return {"configured": True, "sent": False, "error": str(e)}

def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    # Accept "YYYY-MM-DD" or full ISO
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        # fallback for dates without time
        return datetime.fromisoformat(ts + "T00:00:00")

@app.get("/api/models/{model_id}/audit-pack", response_model=dict)
def api_audit_pack(model_id: int, start: Optional[str] = None, end: Optional[str] = None, db: Session = Depends(get_db)):
    m = db.get(ModelDB, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        pack = generate_audit_pack(db, model_id, _parse_iso(start), _parse_iso(end))
        saved_path = write_pack(model_id, pack)  # ← auto-save to backend/var/audit-packs
        out = dict(pack)
        out["_saved_to"] = str(saved_path)
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"audit_pack_failed: {e}")


@app.get("/api/audit-packs", response_model=List[dict])
def list_audit_packs():
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for p in PACK_DIR.glob("*.json"):
        files.append({
            "name": p.name,
            "size_bytes": p.stat().st_size,
            "modified": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
        })
    return sorted(files, key=lambda x: x["name"], reverse=True)

@app.get("/api/audit-packs/{name}")
def download_audit_pack(name: str):
    if ".." in name or not name.endswith(".json"):
        raise HTTPException(status_code=400, detail="Invalid file name")
    p = PACK_DIR / name
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(p), media_type="application/json", filename=p.name)


@app.post("/api/debug/audit-file")
@app.post("/debug/audit-file")
def debug_audit_file():
    event = {"action": "DEBUG_FILE_SINK", "ts": datetime.now(timezone.utc).isoformat()}
    write_event(event)
    return {"ok": True, "dir": str(AUDIT_DIR)}

@app.get("/api/audit-files", response_model=List[dict])
def list_audit_files(user=Depends(require_role("approver", "admin"))):
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for p in AUDIT_DIR.glob("*.jsonl"):
        files.append({
            "name": p.name,
            "size_bytes": p.stat().st_size,
            "modified": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
        })
    # newest first
    return sorted(files, key=lambda x: x["name"], reverse=True)

@app.get("/api/audit-files/{name}")
def download_audit_file(name: str, user=Depends(require_role("approver", "admin"))):
    # simple safety: forbid path traversal and enforce .jsonl
    if ".." in name or not name.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Invalid file name")
    p = AUDIT_DIR / name
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(p), media_type="application/json", filename=p.name)

@app.post("/api/audit-files/purge", response_model=dict)
def purge_audit_files(older_than_days: int = 30, user=Depends(require_role("admin"))):
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    deleted = 0
    for p in AUDIT_DIR.glob("*.jsonl"):
        if datetime.fromtimestamp(p.stat().st_mtime, timezone.utc) < cutoff:
            p.unlink(missing_ok=True)
            deleted += 1
    return {"deleted": deleted, "older_than_days": older_than_days}

@app.get("/api/audit-packs", response_model=List[dict])
def list_audit_packs(
    user=Depends(require_role("approver", "admin")),
):
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for p in PACK_DIR.glob("*.json"):
        files.append({
            "name": p.name,
            "size_bytes": p.stat().st_size,
            "modified": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
        })
    return sorted(files, key=lambda x: x["name"], reverse=True)


@app.get("/api/policy", response_model=dict)
def api_policy(user=Depends(require_role("viewer", "approver", "admin"))):
    return get_policy()

@app.post("/api/policy/reload", response_model=dict)
def api_policy_reload(user=Depends(require_role("admin"))):
    p = reload_policy()
    return {"status": "reloaded", "rules": list((p.get('rules') or {}).keys())}

@app.get("/api/models/{model_id}/policy/check", response_model=dict)
def api_policy_check(model_id: int, target: str, db: Session = Depends(get_db),  user=Depends(require_role("viewer", "approver", "admin"))):
    if target not in ("staging", "production"):
        raise HTTPException(status_code=400, detail="target must be 'staging' or 'production'")
    exp = explain_policy(db, model_id, target)
    return exp

class WaiverIn(BaseModel):
    model_id: int
    rule: str
    target: Optional[str] = None
    reason: str
    requested_by: Optional[str] = None
    approved_by: Optional[str] = None
    expires_at: Optional[str] = None  

class WaiverOut(BaseModel):
    id: int
    model_id: int
    rule: str
    target: Optional[str]
    reason: str
    requested_by: Optional[str]
    approved_by: Optional[str]
    status: str
    expires_at: Optional[datetime] = None
    created_at: datetime
    revoked_at: Optional[datetime] = None
    class Config:
        from_attributes = True

@app.post("/api/waivers", response_model=WaiverOut)
def api_create_waiver(body: WaiverIn, db: Session = Depends(get_db), user=Depends(require_role("admin"))):
    if body.target and body.target not in ("staging", "production"):
        raise HTTPException(status_code=400, detail="target must be 'staging' or 'production'")
    exp = None
    if body.expires_at:
        try:
            exp = datetime.fromisoformat(body.expires_at)
        except Exception:
            raise HTTPException(status_code=400, detail="expires_at must be ISO8601")
    w = create_waiver(db, body.model_id, body.rule, body.reason,
                      body.requested_by, body.approved_by, body.target, exp)
    record_audit(db, "WAIVER_GRANTED", body.model_id, body.approved_by, {"waiver_id": w.id, "rule": w.rule, "target": w.target, "expires_at": body.expires_at})
    return WaiverOut.model_validate(w)

@app.get("/api/waivers", response_model=List[WaiverOut])
def api_list_waivers(model_id: Optional[int] = None, active_only: bool = False, db: Session = Depends(get_db), user=Depends(require_role("approver", "admin"))):
    rows = list_waivers(db, model_id, active_only)
    return [WaiverOut.model_validate(x) for x in rows]

@app.post("/api/waivers/{waiver_id}/revoke", response_model=WaiverOut)
def api_revoke_waiver(waiver_id: int, db: Session = Depends(get_db), user=Depends(require_role("admin"))):
    w = revoke_waiver(db, waiver_id)
    if not w:
        raise HTTPException(status_code=404, detail="Waiver not found")
    record_audit(db, "WAIVER_REVOKED", w.model_id, None, {"waiver_id": w.id})
    return WaiverOut.model_validate(w)

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})

class LoginIn(BaseModel):
    username: str
    role: str  

@app.post("/auth/login")
def auth_login(body: LoginIn):
    role = body.role.lower()
    if role not in ("viewer","approver","admin"):
        raise HTTPException(400, "role must be viewer|approver|admin")
    access = create_access_token(body.username, role)
    refresh = create_refresh_token(body.username, role)
    return {"access_token": access, "refresh_token": refresh, "token_type": "bearer", "expires_in": ACCESS_TTL_MIN * 60, "role": role, "username": body.username}

class RefreshIn(BaseModel):
    refresh_token: str

@app.post("/auth/refresh")
def auth_refresh(body: RefreshIn):
    try:
        data = decode_token(body.refresh_token, expected_type="refresh")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid/expired refresh token")
    new_access = create_access_token(data["sub"], data.get("role","viewer"))
    return {"access_token": new_access, "token_type": "bearer", "expires_in": ACCESS_TTL_MIN * 60}

class KsIn(BaseModel):
    feature: str
    baseline: list[float]
    current: list[float]

@app.post("/api/models/{model_id}/drift/ks")
def api_drift_ks(model_id: int, body: KsIn, db: Session = Depends(get_db),
                 user=Depends(require_role("approver","admin"))):
    return eval_ks(db, model_id, body.feature, body.baseline, body.current)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class DQMissingIn(BaseModel):
    feature: str
    total_count: int = Field(gt=0)
    missing_count: int = Field(ge=0)
    medium_thresh: Optional[float] = None  # e.g., 0.10
    high_thresh: Optional[float] = None    # e.g., 0.20

class DQRangeIn(BaseModel):
    feature: str
    total_count: int = Field(gt=0)
    out_of_range_count: int = Field(ge=0)
    expected_min: Optional[float] = None
    expected_max: Optional[float] = None
    observed_min: Optional[float] = None
    observed_max: Optional[float] = None
    medium_thresh: Optional[float] = None
    high_thresh: Optional[float] = None

@app.post("/api/models/{model_id}/drift/dq/missing")
def api_dq_missing(
    model_id: int,
    body: DQMissingIn,
    db: Session = Depends(get_db),
    user=Depends(require_role("approver","admin")),
):
    mt = body.medium_thresh if body.medium_thresh is not None else 0.10
    ht = body.high_thresh if body.high_thresh is not None else 0.20
    return eval_dq_missing(
        db=db,
        model_id=model_id,
        feature=body.feature,
        total_count=body.total_count,
        missing_count=body.missing_count,
        medium_thresh=mt,
        high_thresh=ht,
    )

@app.post("/api/models/{model_id}/drift/dq/range")
def api_dq_range(
    model_id: int,
    body: DQRangeIn,
    db: Session = Depends(get_db),
    user=Depends(require_role("approver","admin")),
):
    mt = body.medium_thresh if body.medium_thresh is not None else 0.10
    ht = body.high_thresh if body.high_thresh is not None else 0.20
    return eval_dq_range(
        db=db,
        model_id=model_id,
        feature=body.feature,
        total_count=body.total_count,
        out_of_range_count=body.out_of_range_count,
        expected_min=body.expected_min,
        expected_max=body.expected_max,
        observed_min=body.observed_min,
        observed_max=body.observed_max,
        medium_thresh=mt,
        high_thresh=ht,
    )

@app.get("/api/models/{model_id}/status", response_model=dict)
def api_model_status(
    model_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_role("viewer","approver","admin")),
):
    m = db.get(ModelDB, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")

    # Open alerts summary
    open_alerts = (
        db.query(Alert)
        .filter(Alert.model_id == model_id, Alert.status == "open")
        .order_by(Alert.created_at.desc())
        .all()
    )
    by_sev = defaultdict(int)
    by_type = defaultdict(int)
    for a in open_alerts:
        by_sev[a.severity] += 1
        by_type[a.type] += 1

    # Latest metrics per feature/kind
    rows = (
        db.query(MonitoringMetric)
        .filter(MonitoringMetric.model_id == model_id)
        .order_by(MonitoringMetric.created_at.desc())
        .limit(200)
        .all()
    )
    latest = {"psi": {}, "ks": {}, "dq_missing": {}, "dq_range": {}}
    for r in rows:
        kind = None
        if r.method == "psi_eval":
            kind = "psi"
            val = float(r.value) if r.value is not None else None
            sev = (r.details or {}).get("severity")
            payload = {"feature": r.feature, "psi": val, "severity": sev, "at": r.created_at}
        elif r.method == "ks_eval":
            kind = "ks"
            val = float(r.value) if r.value is not None else None
            sev = (r.details or {}).get("severity")
            payload = {"feature": r.feature, "d_stat": val, "severity": sev, "at": r.created_at}
        elif r.method == "dq_missing":
            kind = "dq_missing"
            rate = (r.details or {}).get("missing_rate")
            if rate is None and r.value is not None:
                rate = float(r.value)
            sev = (r.details or {}).get("severity")
            payload = {"feature": r.feature, "missing_rate": rate, "severity": sev, "at": r.created_at}
        elif r.method == "dq_range":
            kind = "dq_range"
            frac = (r.details or {}).get("out_of_range_frac")
            if frac is None and r.value is not None:
                frac = float(r.value)
            sev = (r.details or {}).get("severity")
            payload = {"feature": r.feature, "out_of_range_frac": frac, "severity": sev, "at": r.created_at}
        if kind and r.feature not in latest[kind]:
            latest[kind][r.feature] = payload  # first seen = newest per feature

    # turn maps → lists
    latest_lists = {k: list(v.values()) for k, v in latest.items()}

    return {
        "model": {
            "id": m.id, "name": m.name, "stage": m.stage, "status": m.status,
            "created_at": m.created_at,
        },
        "alerts": {
            "open_total": len(open_alerts),
            "by_severity": dict(by_sev),
            "by_type": dict(by_type),
        },
        "latest_metrics": latest_lists,
    }

@app.get("/api/models/{model_id}/status/badge", response_model=dict)
def api_model_badge(
    model_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_role("viewer","approver","admin")),
):
    m = db.get(ModelDB, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")

    open_alerts = (
        db.query(Alert)
        .filter(Alert.model_id == model_id, Alert.status == "open")
        .all()
    )
    sev_levels = {a.severity for a in open_alerts}
    if "high" in sev_levels:
        level = "critical"
    elif "medium" in sev_levels:
        level = "warning"
    else:
        level = "ok"
    return {
        "model_id": m.id,
        "name": m.name,
        "stage": m.stage,
        "level": level,
        "open_alerts": len(open_alerts),
    }

@app.get("/api/alerts/summary", response_model=dict)
def api_alerts_summary(
    db: Session = Depends(get_db),
    user=Depends(require_role("viewer","approver","admin")),
):
    rows = db.query(Alert).all()
    total = len(rows)
    by_status = defaultdict(int)
    by_type = defaultdict(int)
    by_sev = defaultdict(int)
    for a in rows:
        by_status[a.status] += 1
        by_type[a.type] += 1
        by_sev[a.severity] += 1
    return {
        "total": total,
        "by_status": dict(by_status),
        "by_type": dict(by_type),
        "by_severity": dict(by_sev),
    }

class DQConfigIn(BaseModel):
    feature: str
    missing_medium: Optional[float] = None
    missing_high: Optional[float] = None
    range_medium: Optional[float] = None
    range_high: Optional[float] = None

@app.post("/api/models/{model_id}/drift/dq/config", response_model=dict)
def api_set_dq_config(
    model_id: int,
    body: DQConfigIn,
    db: Session = Depends(get_db),
    user=Depends(require_role("approver","admin")),
):
    m = db.get(ModelDB, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    row = set_dq_thresholds(
        db, model_id, body.feature,
        missing_medium=body.missing_medium, missing_high=body.missing_high,
        range_medium=body.range_medium, range_high=body.range_high,
    )
    return {"config_id": row.id, "feature": body.feature, "message": "DQ thresholds updated"}

# SCHEDULED MONITORING
def _last_metric_time(db: Session, model_id: int, feature: str, method: str) -> datetime:
    row = (
        db.query(MonitoringMetric)
        .filter(
            MonitoringMetric.model_id == model_id,
            MonitoringMetric.feature == feature,
            MonitoringMetric.method == method,
        )
        .order_by(MonitoringMetric.created_at.desc())
        .first()
    )
    return row.created_at if row else datetime.min.replace(tzinfo=None)

def _run_monitor_pass(db: Session):
    # Optionally limit to a stage (e.g., production only)
    q = db.query(ModelDB)
    if MONITOR_ONLY_STAGE:
        q = q.filter(ModelDB.stage == MONITOR_ONLY_STAGE)
    models = q.all()

    for m in models:
        # PSI from snapshots
        baselines = (
            db.query(MonitoringMetric)
            .filter(MonitoringMetric.model_id == m.id, MonitoringMetric.method == "psi_baseline")
            .all()
        )
        for b in baselines:
            last_eval = _last_metric_time(db, m.id, b.feature, "psi_eval")
            snap = (
                db.query(MonitoringMetric)
                .filter(
                    MonitoringMetric.model_id == m.id,
                    MonitoringMetric.feature == b.feature,
                    MonitoringMetric.method == "psi_actual",
                    MonitoringMetric.created_at > last_eval,
                )
                .order_by(MonitoringMetric.created_at.desc())
                .first()
            )
            if snap:
                actual = (snap.details or {}).get("actual") or []
                if actual:
                    try:
                        eval_psi(db, m.id, b.feature, actual)
                    except Exception as e:
                        print(f"[monitor] PSI eval failed for model={m.id} feature={b.feature}: {e}")

        # DQ from snapshots (missing/range)
        # find latest snapshots per feature/kind newer than last eval
        snaps = (
            db.query(MonitoringMetric)
            .filter(MonitoringMetric.model_id == m.id, MonitoringMetric.method == "dq_snapshot")
            .order_by(MonitoringMetric.created_at.desc())
            .all()
        )
        seen = set()
        for s in snaps:
            kind = (s.details or {}).get("kind")
            feat = s.feature
            key = (feat, kind)
            if key in seen:
                continue  # newest-only per feature/kind
            seen.add(key)

            if kind == "missing":
                last_eval = _last_metric_time(db, m.id, feat, "dq_missing")
                if s.created_at > last_eval:
                    tot = (s.details or {}).get("total_count")
                    miss = (s.details or {}).get("missing_count")
                    if isinstance(tot, int) and isinstance(miss, int):
                        try:
                            from app.crud.monitoring import eval_dq_missing
                            eval_dq_missing(db, m.id, feat, tot, miss)
                        except Exception as e:
                            print(f"[monitor] DQ missing eval failed m={m.id} f={feat}: {e}")

            elif kind == "range":
                last_eval = _last_metric_time(db, m.id, feat, "dq_range")
                if s.created_at > last_eval:
                    tot = (s.details or {}).get("total_count")
                    oor = (s.details or {}).get("out_of_range_count")
                    if isinstance(tot, int) and isinstance(oor, int):
                        try:
                            from app.crud.monitoring import eval_dq_range
                            emn = (s.details or {}).get("expected_min")
                            emx = (s.details or {}).get("expected_max")
                            omn = (s.details or {}).get("observed_min")
                            omx = (s.details or {}).get("observed_max")
                            eval_dq_range(db, m.id, feat, tot, oor, emn, emx, omn, omx)
                        except Exception as e:
                            print(f"[monitor] DQ range eval failed m={m.id} f={feat}: {e}")

@app.post("/admin/monitor/run", response_model=dict)
def admin_monitor_run(user=Depends(require_role("admin"))):
    _monitor_once()
    return {"ran": True, "stage_filter": MONITOR_ONLY_STAGE, "interval_sec": MONITOR_INTERVAL_SEC}

def _monitor_once():
    db = SessionLocal()
    try:
        _run_monitor_pass(db)
    finally:
        db.close()

async def _monitor_loop():
    while True:
        try:
            _monitor_once()
        except Exception as e:
            print(f"[monitor] pass error: {e}")
        await asyncio.sleep(MONITOR_INTERVAL_SEC)

@app.get("/admin", include_in_schema=False)
def admin_page():
    return FileResponse(STATIC_DIR / "admin.html")

@app.get("/public/healthz", include_in_schema=False)
def public_healthz(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        return Response(content='{"status":"error"}', media_type="application/json", status_code=503)

@app.get("/public/version", include_in_schema=False)
def public_version():
    return {"name": "mlops-governance-api", "version": "0.2.0"}

@app.get("/public/status", response_model=list[dict], include_in_schema=False)
def public_status(db: Session = Depends(get_db)):
    """
    Public read-only summary per model:
      - id, name, stage
      - open_alerts count
      - last_metric_at (UTC ISO)
    """
    models = db.query(ModelDB).order_by(ModelDB.id.asc()).all()
    out = []
    for m in models:
        open_cnt = db.query(Alert).filter(Alert.model_id == m.id, Alert.status == "open").count()
        last_metric = (
            db.query(func.max(MonitoringMetric.created_at))
            .filter(MonitoringMetric.model_id == m.id)
            .scalar()
        )
        out.append({
            "id": m.id,
            "name": m.name,
            "stage": m.stage,
            "open_alerts": int(open_cnt),
            "last_metric_at": last_metric.isoformat() if last_metric else None,
        })
    return out

@app.get("/badge/model/{model_id}.svg", include_in_schema=False)
def model_badge(model_id: int, db: Session = Depends(get_db)):
    m = db.get(ModelDB, model_id)
    if not m:
        return Response(content="<svg xmlns='http://www.w3.org/2000/svg' width='120' height='20'></svg>", media_type="image/svg+xml", status_code=404)
    open_cnt = db.query(Alert).filter(Alert.model_id == model_id, Alert.status == "open").count()
    label = f"{m.name[:12]}".replace("&","&amp;")
    if open_cnt == 0:
        msg, fill = "OK", "#10b981"  # green
    else:
        msg, fill = f"ALERTS {open_cnt}", "#ef4444"  # red

    # a simple shields-style badge
    left = label
    right = msg
    # rough widths
    lw = max(60, 7 * len(left) + 20)
    rw = max(60, 8 * len(right) + 20)
    w = lw + rw
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="20" role="img" aria-label="{left}: {right}">
  <linearGradient id="s" x2="0" y2="100%"><stop offset="0" stop-color="#fff" stop-opacity=".7"/><stop offset=".1" stop-opacity=".1"/><stop offset=".9" stop-opacity=".3"/><stop offset="1" stop-opacity=".5"/></linearGradient>
  <mask id="m"><rect width="{w}" height="20" rx="3" fill="#fff"/></mask>
  <g mask="url(#m)">
    <rect width="{lw}" height="20" fill="#555"/>
    <rect x="{lw}" width="{rw}" height="20" fill="{fill}"/>
    <rect width="{w}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle"
     font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="{lw/2}" y="14">{left}</text>
    <text x="{lw+rw/2}" y="14">{right}</text>
  </g>
</svg>"""
    return Response(content=svg, media_type="image/svg+xml")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

