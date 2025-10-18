from __future__ import annotations
from datetime import datetime
from typing import List, Optional
from sqlalchemy.orm import Session
from app.models.waiver import Waiver

def create_waiver(
    db: Session,
    model_id: int,
    rule: str,
    reason: str,
    requested_by: Optional[str],
    approved_by: Optional[str],
    target: Optional[str] = None,
    expires_at: Optional[datetime] = None,
) -> Waiver:
    w = Waiver(
        model_id=model_id, rule=rule, target=target, reason=reason,
        requested_by=requested_by, approved_by=approved_by,
        expires_at=expires_at, status="approved"
    )
    db.add(w); db.commit(); db.refresh(w)
    return w

def revoke_waiver(db: Session, waiver_id: int) -> Optional[Waiver]:
    w = db.get(Waiver, waiver_id)
    if not w:
        return None
    w.status = "revoked"
    w.revoked_at = datetime.utcnow()
    db.commit(); db.refresh(w)
    return w

def list_waivers(db: Session, model_id: Optional[int] = None, active_only: bool = False) -> List[Waiver]:
    q = db.query(Waiver)
    if model_id is not None:
        q = q.filter(Waiver.model_id == model_id)
    if active_only:
        now = datetime.utcnow()
        q = q.filter(Waiver.status == "approved").filter(
            (Waiver.expires_at == None) | (Waiver.expires_at >= now)  # noqa: E711
        )
    return q.order_by(Waiver.created_at.desc()).all()
