# ==============================================================================
# LangGraph API — Dockerfile
# Ejecuta el servidor de desarrollo de LangGraph dentro de un contenedor.
# El grafo se importa desde el código del proyecto montado en /app.
# ==============================================================================

FROM python:3.12-slim AS base

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Instalar dependencias Python ───────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    # langgraph-cli incluye el servidor de desarrollo
    && pip install --no-cache-dir "langgraph-cli[inmem]>=0.1.0" \
    # fastmcp es requerido por el MCP server de Redmine (ver AGENTS.md)
    && pip install --no-cache-dir "fastmcp>=2.0"

# ── Copiar código fuente ───────────────────────────────────────────────────
COPY . .

# Puerto del servidor de LangGraph
EXPOSE 8123

HEALTHCHECK --interval=30s --timeout=15s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8123/ok || exit 1

# Arrancar el servidor de desarrollo de LangGraph
# --host 0.0.0.0 permite conexiones desde otros contenedores en la red Docker
# --no-browser evita intentar abrir el navegador (innecesario en contenedor)
CMD ["langgraph", "dev", "--host", "localhost", "--port", "8123", "--no-browser"]
