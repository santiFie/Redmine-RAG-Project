"""
tests/unit/redmine/test_client.py
=================================
Pruebas unitarias para RedmineClient mockeando la red con pytest-httpx.
"""

import pytest
from pytest_httpx import HTTPXMock

from src.redmine.client import (
    RedmineClient,
    RedmineNotFoundError,
    RedmineValidationError,
)


@pytest.fixture
def client():
    """Fixture que devuelve un cliente Redmine con configuración de prueba."""
    return RedmineClient(base_url="http://test-redmine.local", api_key="test-key-123")


def test_client_init_sin_vars(monkeypatch):
    """Prueba que falla si no hay URL o API key."""
    monkeypatch.delenv("REDMINE_URL", raising=False)
    monkeypatch.delenv("REDMINE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="REDMINE_URL no configurada"):
        RedmineClient(base_url=None, api_key="key")

    with pytest.raises(ValueError, match="REDMINE_API_KEY no configurada"):
        RedmineClient(base_url="http://test", api_key=None)


def test_get_issue_success(client: RedmineClient, httpx_mock: HTTPXMock):
    """Prueba que get_issue parsea la respuesta HTTP exitosa."""
    httpx_mock.add_response(
        method="GET",
        url="http://test-redmine.local/issues/42.json",
        json={"issue": {"id": 42, "subject": "Test Issue"}},
    )

    issue = client.get_issue(42)
    assert issue["id"] == 42
    assert issue["subject"] == "Test Issue"


def test_get_issue_not_found(client: RedmineClient, httpx_mock: HTTPXMock):
    """Prueba que levanta RedmineNotFoundError en 404."""
    httpx_mock.add_response(
        method="GET", url="http://test-redmine.local/issues/999.json", status_code=404
    )

    with pytest.raises(RedmineNotFoundError):
        client.get_issue(999)


def test_create_issue_validation_error(client: RedmineClient, httpx_mock: HTTPXMock):
    """Prueba que levanta RedmineValidationError en 422 con detalles."""
    httpx_mock.add_response(
        method="POST",
        url="http://test-redmine.local/issues.json",
        status_code=422,
        json={"errors": ["Subject can't be blank", "Project is required"]},
    )

    with pytest.raises(RedmineValidationError) as exc_info:
        client.create_issue(project_id="core", subject="")

    assert "Subject can't be blank" in exc_info.value.errors
    assert "Project is required" in exc_info.value.errors


def test_auth_headers_enviados(client: RedmineClient, httpx_mock: HTTPXMock):
    """Verifica que la API Key se envía en los headers."""
    httpx_mock.add_response(
        method="GET",
        url="http://test-redmine.local/issues.json?status_id=open&limit=25&offset=0",
        json={"issues": [], "total_count": 0},
    )

    client.list_issues()

    request = httpx_mock.get_request()
    assert request is not None
    assert request.headers.get("x-redmine-api-key") == "test-key-123"
