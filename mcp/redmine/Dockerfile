# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# Metadata
LABEL org.opencontainers.image.title="redmine-mcp-server"
LABEL org.opencontainers.image.description="FastMCP stdio server para la API REST de Redmine"

# No pyc files, no buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Dependencias ──────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Código ────────────────────────────────────────────────────────────────────
COPY client.py .
COPY server.py .

# ── Entrypoint ────────────────────────────────────────────────────────────────
# El servidor se lanza en modo stdio: el host MCP (Claude Desktop/Code)
# arranca el contenedor y se comunica vía stdin/stdout.
CMD ["python", "server.py"]
