# app/crud/approval.py
from typing import Any, Dict, List, Set, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from app.models.approval import ApprovalRequest
from app.models.model import Model
from app.services.audit import record_audit
from app.utils.policy import explain_policy 
from app.metrics import policy_blocks_total

try:
    from main import POLICY_BLOCKS
except Exception:
    POLICY_BLOCKS = None

VALID_TARGETS = {"staging", "production"}
VALID_DECISIONS = {"approved", "rejected"}

# Quorum per target
MIN_UNIQUE_APPROVERS = {"staging": 1, "production": 2}

# Governance knobs
ALLOW_REQUESTER_TO_APPROVE = False
ALLOW_DUPLICATE_DECISIONS = False


def request_promotion(db: Session, model_id: int, target: str, justification: str, requested_by: str) -> ApprovalRequest:
    """Create a promotion approval request."""
    target = (target or "").strip().lower()
    if target not in VALID_TARGETS:
        raise ValueError(f"Invalid target '{target}'. Must be one of {sorted(VALID_TARGETS)}.")

    m = db.get(Model, model_id)
    if not m:
        raise ValueError(f"Model {model_id} not found")

    try:
        if getattr(m, "stage", None) == target:
            raise ValueError(f"Model already in '{target}'")
    except Exception:
        pass

    req = ApprovalRequest(
        model_id=model_id,
        target=target,
        justification=justification,
        requested_by=requested_by,
    )
    db.add(req)
    db.commit()
    db.refresh(req)

    record_audit(
        db,
        "REQUEST_PROMOTION",
        model_id,
        requested_by,
        {"target": target, "justification": justification, "approval_id": req.id},
    )
    return req


def add_decision(db: Session, approval_id: int, decided_by: str, decision: str, note: Optional[str] = None) -> ApprovalRequest:
    """Record an approval/rejection; on quorum, enforce policy, then flip stage."""
    req = db.get(ApprovalRequest, approval_id)
    if not req:
        raise ValueError(f"Approval {approval_id} not found")

    # Terminal states: do nothing
    if req.status in ("approved", "rejected", "blocked", "completed"):
        return req

    d = (decision or "").strip().lower()
    if d not in VALID_DECISIONS:
        raise ValueError(f"Invalid decision '{decision}'. Must be one of {sorted(VALID_DECISIONS)}.")

    # SoD: requester cannot approve their own request (unless allowed)
    if not ALLOW_REQUESTER_TO_APPROVE and req.requested_by and decided_by.lower() == str(req.requested_by).lower():
        raise ValueError("Requester cannot approve their own promotion request")

    # Prevent duplicate decisions by the same reviewer (unless allowed)
    prev_by: Set[str] = {str(x.get("by", "")).lower() for x in (req.decisions or [])}
    if not ALLOW_DUPLICATE_DECISIONS and decided_by.lower() in prev_by:
        raise ValueError(f"Reviewer '{decided_by}' has already recorded a decision for this request")

    # Append this decision
    decisions: List[Dict[str, Any]] = list(req.decisions or [])
    decisions.append(
        {
            "by": decided_by,
            "decision": d,
            "note": note or "",
            "at": datetime.utcnow().isoformat(),
        }
    )
    req.decisions = decisions

    # Immediate reject path
    if d == "rejected":
        req.status = "rejected"
        db.commit()
        db.refresh(req)
        record_audit(
            db,
            "DECIDE_PROMOTION",
            req.model_id,
            decided_by,
            {"approval_id": approval_id, "decision": "rejected", "note": note or ""},
        )
        return req

    # Count unique approvals towards quorum
    approved_by: Set[str] = {str(x["by"]).lower() for x in decisions if x.get("decision") == "approved"}
    needed = MIN_UNIQUE_APPROVERS.get(req.target, 1)
    approvals_count = len(approved_by)

    # Log the approval decision (may still be pending)
    record_audit(
        db,
        "DECIDE_PROMOTION",
        req.model_id,
        decided_by,
        {
            "approval_id": approval_id,
            "decision": "approved",
            "note": note or "",
            "approvals": approvals_count,
            "needed": needed,
        },
    )

    if approvals_count < needed:
        # Still pending; wait for more approvals
        req.status = "pending"
        db.commit()
        db.refresh(req)
        return req

    # === Quorum satisfied → POLICY GATE (before changing the stage) ===
    record_audit(
        db,
        "APPROVAL_QUORUM_MET",
        req.model_id,
        decided_by,
        {"approval_id": approval_id, "target": req.target, "approvals": approvals_count},
    )

    pol = explain_policy(db, req.model_id, req.target)
    if not pol.get("allow", False):
        # Distinguish policy block from human rejection
        req.status = "blocked"
        db.commit()
        db.refresh(req)
        try:
            if POLICY_BLOCKS:
                POLICY_BLOCKS.inc()
        except Exception:
            pass
        record_audit(
            db,
            "POLICY_BLOCK",
            req.model_id,
            decided_by,
            {"approval_id": approval_id, "blocks": pol.get("blocks", []), "target": req.target},
        )
        policy_blocks_total.inc()
        return req

    # Passed policy → flip stage
    model = db.get(Model, req.model_id)
    if model:
        try:
            model.stage = req.target  # if Enum, your Model setter can coerce
        except Exception:
            model.stage = req.target
        try:
            setattr(model, "updated_at", datetime.utcnow())
        except Exception:
            pass

    req.status = "approved"
    db.commit()
    db.refresh(req)

    record_audit(
        db,
        "STAGE_CHANGED",
        model.id if model else None,
        decided_by,
        {"new_stage": getattr(model, "stage", req.target), "approval_id": approval_id},
    )
    return req
