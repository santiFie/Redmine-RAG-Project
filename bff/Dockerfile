# ==============================================================================
# BFF — Dockerfile
# Multi-stage build: instala dependencias y ejecuta con uvicorn
# ==============================================================================

FROM python:3.12-slim AS base

WORKDIR /app

# Instalar dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Stage: dependencias ────────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage: producción ──────────────────────────────────────────────────────
FROM deps AS production

COPY app/ ./app/

# Variables de entorno por defecto (sobrescribir en .env o docker-compose)
ENV PORT=8000
ENV PYTHONPATH=/app

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 2"]
