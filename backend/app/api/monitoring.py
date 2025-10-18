from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.deps.auth import require_role
from app.crud.monitoring import eval_dq_missing, eval_dq_range

router = APIRouter()

class DQMissingIn(BaseModel):
    feature: str
    total_count: int = Field(gt=0)
    missing_count: int = Field(ge=0)
    medium_thresh: float | None = Field(default=None, description="Optional override, e.g. 0.10")
    high_thresh: float | None = Field(default=None, description="Optional override, e.g. 0.20")

@router.post("/api/models/{model_id}/drift/dq/missing")
def dq_missing(model_id: int, body: DQMissingIn,
               db: Session = Depends(get_db), user=Depends(require_role("approver"))):
    mt = body.medium_thresh if body.medium_thresh is not None else 0.10
    ht = body.high_thresh if body.high_thresh is not None else 0.20
    return eval_dq_missing(db, model_id, body.feature, body.total_count, body.missing_count, mt, ht)


class DQRangeIn(BaseModel):
    feature: str
    total_count: int = Field(gt=0)
    out_of_range_count: int = Field(ge=0)
    expected_min: float | None = None
    expected_max: float | None = None
    observed_min: float | None = None
    observed_max: float | None = None
    medium_thresh: float | None = Field(default=None)
    high_thresh: float | None = Field(default=None)

@router.post("/api/models/{model_id}/drift/dq/range")
def dq_range(model_id: int, body: DQRangeIn,
             db: Session = Depends(get_db), user=Depends(require_role("approver"))):
    mt = body.medium_thresh if body.medium_thresh is not None else 0.10
    ht = body.high_thresh if body.high_thresh is not None else 0.20
    return eval_dq_range(
        db, model_id, body.feature, body.total_count, body.out_of_range_count,
        body.expected_min, body.expected_max, body.observed_min, body.observed_max,
        mt, ht
    )
