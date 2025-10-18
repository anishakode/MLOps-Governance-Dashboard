from __future__ import annotations
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from datetime import datetime
from app.core.database import Base

class MonitoringMetric(Base):
    __tablename__ = "monitoring_metrics"
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("models.id"), index=True, nullable=False)
    feature = Column(String(255), index=True, nullable=False)
    method = Column(String(64), nullable=False)  # "psi_baseline" | "psi_eval"
    value = Column(String(64), nullable=True)    # store PSI float as string for simplicity
    details = Column(JSON, default=dict)         # {"expected":[...], "actual":[...]}
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, index=True, nullable=False)
    type = Column(String(64), nullable=False)        # "drift"
    severity = Column(String(16), nullable=False)    # low | medium | high
    message = Column(String(1000), nullable=False)
    details = Column(JSON, default=dict)
    status = Column(String(16), default="open")      # open | resolved
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    resolved_at = Column(DateTime, nullable=True)
