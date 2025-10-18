from typing import Any, Dict, List, Set
from sqlalchemy.orm import Session
from datetime import datetime
from app.models.approval import ApprovalRequest
from app.models.model import Model
from app.services.audit import record_audit
from app.utils.policy import evaluate_promotion_policy

VALID_TARGETS = {"staging", "production"}
VALID_DECISIONS = {"approved", "rejected"}

MIN_UNIQUE_APPROVERS = {"staging": 1, "production": 2}
ALLOW_REQUESTER_TO_APPROVE = False
ALLOW_DUPLICATE_DECISIONS = False

def request_promotion(db: Session, model_id: int, target: str, justification: str, requested_by: str):
    target = (target or "").strip().lower()
    if target not in VALID_TARGETS:
        raise ValueError(f"Invalid target '{target}'. Must be one of {sorted(VALID_TARGETS)}.")

    m = db.get(Model, model_id)
    if not m:
        raise ValueError(f"Model {model_id} not found")

    req = ApprovalRequest(model_id=model_id, target=target, justification=justification, requested_by=requested_by)
    db.add(req)
    db.commit()
    db.refresh(req)
    record_audit(db, "REQUEST_PROMOTION", model_id, requested_by,
             {"target": target, "justification": justification, "approval_id": req.id})
    return req

def add_decision(db: Session, approval_id: int, decided_by: str, decision: str, note: str | None = None):
    req = db.get(ApprovalRequest, approval_id)
    if not req:
        raise ValueError(f"Approval {approval_id} not found")
    
    if req.status != "pending":
        return req
    
    d = (decision or "").strip().lower()
    if d not in VALID_DECISIONS:
        raise ValueError(f"Invalid decision '{decision}'. Must be one of {sorted(VALID_DECISIONS)}.")

    # SoD: requester cannot approve own request
    if not ALLOW_REQUESTER_TO_APPROVE and req.requested_by and decided_by.lower() == str(req.requested_by).lower():
        raise ValueError("Requester cannot approve their own promotion request")

    # Prevent duplicate decisions by same user (unless allowed)
    prev_by: Set[str] = {str(x.get("by", "")).lower() for x in (req.decisions or [])}
    if not ALLOW_DUPLICATE_DECISIONS and decided_by.lower() in prev_by:
        raise ValueError(f"Reviewer '{decided_by}' has already recorded a decision for this request")

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

   # If any reviewer explicitly rejects → request becomes rejected immediately
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

    # Count unique approvals for quorum
    approved_by: Set[str] = {str(x["by"]).lower() for x in decisions if x.get("decision") == "approved"}
    needed = MIN_UNIQUE_APPROVERS[req.target]
    approvals_count = len(approved_by)

    # Record the decision event
    record_audit(
        db,
        "DECIDE_PROMOTION",
        req.model_id,
        decided_by,
        {"approval_id": approval_id, "decision": "approved", "note": note or "", "approvals": approvals_count, "needed": needed},
    )

    if approvals_count < needed:
        # Still pending; wait for more approvals
        db.commit()
        db.refresh(req)
        return req

    # === Quorum satisfied → POLICY GATE ===
    evalr = evaluate_promotion_policy(db, req.model_id, req.target)
    if not evalr["pass"]:
        req.status = "rejected"
        db.commit()
        db.refresh(req)

        record_audit(
            db,
            "POLICY_BLOCKED",
            req.model_id,
            decided_by,
            {"approval_id": approval_id, "reasons": evalr["failures"], "target": req.target, "policy_hash": evalr.get("policy_hash")},
        )
        return req

    # Passed policy (maybe via waivers)
    if evalr.get("waivers_used"):
        record_audit(db, "POLICY_WAIVER_USED", req.model_id, decided_by,
                     {"approval_id": approval_id, "waivers": evalr["waivers_used"], "target": req.target, "policy_hash": evalr.get("policy_hash")})

    # Finalize and flip stage
    req.status = "approved"
    model = db.get(Model, req.model_id)
    if model:
        model.stage = req.target
        try:
            setattr(model, "updated_at", datetime.utcnow())
        except Exception:
            pass

    db.commit()
    db.refresh(req)

    record_audit(
        db,
        "STAGE_CHANGED",
        model.id if model else None,
        decided_by,
        {"new_stage": model.stage, "approval_id": approval_id},
    )
    return req