# from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.sql import func
# from datetime import datetime
# import enum

# Base = declarative_base()

# class ModelStage(str, enum.Enum):
#     DEVELOPMENT = "development"
#     STAGING = "staging" 
#     PRODUCTION = "production"
#     ARCHIVED = "archived"

# class ModelStatus(str, enum.Enum):
#     ACTIVE = "active"
#     INACTIVE = "inactive"
#     DEPRECATED = "deprecated"

# class Model(Base):
#     __tablename__ = "models"
    
#     id = Column(Integer, primary_key=True, index=True)
#     name = Column(String(255), unique=True, index=True, nullable=False)
#     description = Column(Text, nullable=True)
#     version = Column(String(50), nullable=False, default="1.0.0")
#     stage = Column(String(50), default=ModelStage.DEVELOPMENT, nullable=False)
#     status = Column(String(50), default=ModelStatus.ACTIVE, nullable=False)
#     model_metadata = Column(JSON, default=dict)  # CHANGED: metadata -> model_metadata
#     created_at = Column(DateTime(timezone=True), server_default=func.now())
#     updated_at = Column(DateTime(timezone=True), onupdate=func.now())

# class ModelVersion(Base):
#     __tablename__ = "model_versions"
    
#     id = Column(Integer, primary_key=True, index=True)
#     model_id = Column(Integer, nullable=False, index=True)
#     version_number = Column(String(50), nullable=False)
#     experiment_id = Column(String(255), nullable=True)
#     run_id = Column(String(255), nullable=True)
#     artifacts_path = Column(Text, nullable=True)
#     metrics = Column(JSON, default=dict)
#     parameters = Column(JSON, default=dict)
#     created_at = Column(DateTime(timezone=True), server_default=func.now())


from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from datetime import datetime
import enum

from app.core.database import Base 

class ModelStage(str, enum.Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

class ModelStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"

class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    version = Column(String(50), nullable=False, default="1.0.0")
    stage = Column(String(50), default=ModelStage.DEVELOPMENT, nullable=False)
    status = Column(String(50), default=ModelStatus.ACTIVE, nullable=False)
    model_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
