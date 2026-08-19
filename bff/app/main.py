"""
bff/app/main.py
===============
Entrypoint del Backend for Frontend (BFF) — FastAPI.

Responsabilidades:
  - Configuración de CORS (orígenes desde variable de entorno).
  - Montaje de routers: autenticación y chat SSE.
  - Health check endpoint.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth.router import router as auth_router
from app.chat.router import router as chat_router
from app.config import settings

# ---------------------------------------------------------------------------
# Instancia principal de la aplicación
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agente Redmine + RAG — BFF",
    description=(
        "Backend for Frontend que gestiona autenticación JWT y actúa como "
        "proxy de streaming SSE hacia LangGraph Platform."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Middleware CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Thread-ID"],  # Exponer el thread_id en el header de respuesta
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(auth_router)
app.include_router(chat_router)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Sistema"], summary="Estado del servicio")
async def health() -> dict:
    """Endpoint de health check para load balancers y Docker healthcheck."""
    return {
        "status": "ok",
        "service": "bff",
        "langgraph_url": settings.langgraph_api_url,
    }
