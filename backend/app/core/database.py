# from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# import os

# # Database URL
# SQLALCHEMY_DATABASE_URL = "postgresql://mlops_user:mlops_pass@localhost:5432/mlops_governance"

# # Create engine
# engine = create_engine(SQLALCHEMY_DATABASE_URL)

# # Create session factory
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# # Base class for models
# Base = declarative_base()

# # Dependency injection
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()


# from sqlalchemy import create_engine, text
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# import time

# # Docker PostgreSQL - with connection pooling and retry logic
# SQLALCHEMY_DATABASE_URL = "postgresql://mlops_user:mlops_pass@localhost:5432/mlops_governance"

# # Create engine with connection pooling and retry
# engine = create_engine(
#     SQLALCHEMY_DATABASE_URL,
#     pool_pre_ping=True,  # Test connections before using
#     pool_recycle=300,    # Recycle connections every 5 minutes
#     echo=True,           # Log SQL for debugging
# )

# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# def get_db():
#     """Get database session with retry logic"""
#     max_retries = 3
#     for attempt in range(max_retries):
#         try:
#             db = SessionLocal()
#             # Test connection
#             db.execute(text("SELECT 1"))
#             return db
#         except Exception as e:
#             if attempt == max_retries - 1:
#                 raise e
#             time.sleep(0.1)  # Brief pause before retry
#         finally:
#             if 'db' in locals():
#                 db.close()


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# If you're using Docker Postgres:
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://mlops_user:mlops_pass@localhost:5432/mlops_governance"
)

engine = create_engine(SQLALCHEMY_DATABASE_URL, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Single source of truth for Base
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
