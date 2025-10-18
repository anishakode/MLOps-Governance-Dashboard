# Dockerfile (repo root)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps from root requirements.txt
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code into image
COPY backend/ ./backend

# Work inside your backend folder (where main.py, alembic.ini live)
WORKDIR /app/backend

# Make sure Alembic knows where its config is
ENV ALEMBIC_CONFIG=/app/backend/alembic.ini
ENV PYTHONUNBUFFERED=1

# Run migrations, then start API
CMD ["bash","-lc","alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port 8000"]
