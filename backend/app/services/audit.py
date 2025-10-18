from __future__ import annotations
from typing import Optional, Dict, Any
from datetime import datetime

from sqlalchemy.orm import Session

from app.models.audit import AuditLog
from app.utils.audit_sink import write_event

def record_audit(
    db: Session,
    action: str,
    model_id: Optional[int],
    actor: Optional[str],
    details: Dict[str, Any],
) -> AuditLog:
    """
    Persist audit to DB and mirror to filesystem as JSONL.
    """
    row = AuditLog(action=action, model_id=model_id, actor=actor, details=details)
    db.add(row)
    db.commit()
    db.refresh(row)

    write_event({
        "id": row.id,
        "action": row.action,
        "model_id": row.model_id,
        "actor": row.actor,
        "details": row.details or {},
        "created_at": row.created_at.isoformat() if row.created_at else datetime.utcnow().isoformat(),
    })
    return row
