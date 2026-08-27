"""
tests/smoke/test_services_health.py
===================================
Pruebas de humo (Smoke Tests) para validar que los servicios básicos
del stack de Docker están arriba y respondiendo correctamente HTTP.
Diseñado especialmente para el paso final del pipeline de CI.
"""

import httpx
import pytest

# Definición de URLs por defecto de los servicios locales
BFF_URL = "http://localhost:8000"
LANGGRAPH_API_URL = "http://localhost:8123"
REDMINE_URL = "http://localhost:3000"
QDRANT_URL = "http://localhost:6333"

# Opciones compartidas de timeout (no queremos colgar la CI)
TIMEOUT = 5.0


@pytest.mark.smoke
def test_bff_healthcheck():
    """Valida el endpoint de salud del Backend for Frontend."""
    try:
        response = httpx.get(f"{BFF_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200, f"BFF retornó status {response.status_code}"
        data = response.json()
        assert data.get("status") == "ok", "BFF status no es 'ok'"
    except httpx.RequestError as e:
        pytest.fail(f"Falla de conexión al BFF en {BFF_URL}: {e}")


@pytest.mark.smoke
def test_langgraph_api_healthcheck():
    """Valida el endpoint de salud nativo de LangGraph API."""
    try:
        response = httpx.get(f"{LANGGRAPH_API_URL}/ok", timeout=TIMEOUT)
        assert response.status_code == 200, f"LangGraph API retornó status {response.status_code}"
    except httpx.RequestError as e:
        pytest.fail(f"Falla de conexión a LangGraph API en {LANGGRAPH_API_URL}: {e}")


@pytest.mark.smoke
def test_redmine_healthcheck():
    """Valida que el contenedor de Redmine esté respondiendo peticiones web."""
    try:
        response = httpx.get(f"{REDMINE_URL}/", timeout=TIMEOUT)
        # Redmine puede devolver 200 o 302 (redirección al login), ambas indican que está vivo.
        assert response.status_code in (200, 302), f"Redmine retornó status {response.status_code}"
    except httpx.RequestError as e:
        pytest.fail(f"Falla de conexión a Redmine en {REDMINE_URL}: {e}")


@pytest.mark.smoke
def test_qdrant_healthcheck():
    """Valida que Qdrant esté listo para aceptar tráfico."""
    try:
        # Endpoint de readiness de Qdrant
        response = httpx.get(f"{QDRANT_URL}/readyz", timeout=TIMEOUT)
        assert response.status_code == 200, f"Qdrant retornó status {response.status_code}"
    except httpx.RequestError as e:
        pytest.fail(f"Falla de conexión a Qdrant en {QDRANT_URL}: {e}")
