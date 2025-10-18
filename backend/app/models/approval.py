from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from datetime import datetime
from app.core.database import Base

class ApprovalRequest(Base):
    __tablename__ = "approval_requests"
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False, index=True)
    target = Column(String(32), nullable=False)          # "staging" | "production"
    justification = Column(String(1000))
    status = Column(String(32), default="pending")       # "pending" | "approved" | "rejected"
    requested_by = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    decisions = Column(JSON, default=list)               # [{"by":..., "decision":"approved|rejected", "note":..., "at":...}]
