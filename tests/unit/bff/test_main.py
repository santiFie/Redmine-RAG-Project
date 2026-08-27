"""
tests/unit/bff/test_main.py
===========================
Pruebas unitarias para los endpoints estáticos del Backend for Frontend (FastAPI).
"""

import sys
from pathlib import Path

# Agregar el directorio raíz o bff al PYTHONPATH para resolver `app`
_bff_dir = Path(__file__).resolve().parent.parent.parent.parent / "bff"
sys.path.insert(0, str(_bff_dir))

from app.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)


def test_health_check_endpoint():
    """Valida que el endpoint /health devuelve 200 y el formato JSON esperado."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "bff"
    # langgraph_url vendrá de las variables de entorno, por lo que comprobamos que la clave exista
    assert "langgraph_url" in data
