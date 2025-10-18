from __future__ import annotations
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from app.core.database import Base

class Waiver(Base):
    __tablename__ = "waivers"
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("models.id"), index=True, nullable=False)
    rule = Column(String(128), nullable=False)         # e.g., "block_on_alerts", "require_baselines"
    target = Column(String(32), nullable=True)         # "staging" | "production" | None (any)
    reason = Column(Text, nullable=False)
    requested_by = Column(String(128), nullable=True)
    approved_by = Column(String(128), nullable=True)
    status = Column(String(16), default="approved")    # "approved" | "revoked"
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
