# ─────────────────────────────────────────────
# Dockerfile — Spaceship Titanic MLOps Pipeline
# Phase 4: Dockerize
# ─────────────────────────────────────────────

# Base image — slim Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# ── System dependencies (needed for XGBoost / LightGBM on Linux) ──────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy source code ───────────────────────────────────────────────────────────
COPY src/ ./src/

# ── Copy trained model artifact ────────────────────────────────────────────────
COPY models/ ./models/

# ── Copy processed data (needed for feature engineering reference) ─────────────
COPY data/processed/ ./data/processed/

# ── Set PYTHONPATH so imports work correctly ───────────────────────────────────
ENV PYTHONPATH=/app/src

# ── Expose FastAPI port ────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# ── Start FastAPI server ───────────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
