from sqlalchemy import Column, Integer, String, DateTime, JSON
from datetime import datetime
from app.core.database import Base

class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True)
    action = Column(String(128), index=True)             # e.g., REQUEST_PROMOTION, DECIDE_PROMOTION, STAGE_CHANGED
    model_id = Column(Integer, index=True, nullable=True)
    actor = Column(String(255), nullable=True)
    details = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
