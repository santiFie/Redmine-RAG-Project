"""
tests/unit/redmine/test_schemas.py
==================================
Pruebas unitarias para validar que los esquemas Pydantic transforman correctamente
los JSON de entrada y arrojan excepciones ante datos inválidos.
"""

import pytest
from pydantic import ValidationError

from src.redmine.schemas import RedmineIssue, RedmineProject, RedmineUser


def test_redmine_user_schema_valido():
    data = {"id": 1, "name": "Admin User"}
    user = RedmineUser(**data)
    assert user.id == 1
    assert user.name == "Admin User"


def test_redmine_user_schema_invalido():
    data = {"id": "no-es-int", "name": "Admin User"}
    with pytest.raises(ValidationError):
        RedmineUser(**data)


def test_redmine_project_schema():
    data = {"id": 42, "identifier": "proyecto-test", "name": "Proyecto Test"}
    project = RedmineProject(**data)
    assert project.id == 42
    assert project.identifier == "proyecto-test"
    assert project.name == "Proyecto Test"
    assert project.description is None


def test_redmine_issue_schema_completo():
    data = {
        "id": 100,
        "subject": "Error en login",
        "project": {"id": 1, "identifier": "core", "name": "Core"},
        "status": {"id": 1, "name": "Nueva"},
        "done_ratio": 50,
        "journals": [
            {
                "id": 200,
                "user": {"id": 2, "name": "Dev User"},
                "notes": "Revisando el error",
                "created_on": "2023-10-01T12:00:00Z",
            }
        ],
    }
    issue = RedmineIssue(**data)
    assert issue.id == 100
    assert issue.subject == "Error en login"
    assert issue.project is not None
    assert issue.project.name == "Core"
    assert issue.status is not None
    assert issue.status.name == "Nueva"
    assert issue.done_ratio == 50
    assert len(issue.journals) == 1
    assert issue.journals[0].notes == "Revisando el error"

    # Probar to_document_text() sin que rompa
    doc_text = issue.to_document_text()
    assert "ID: 100" in doc_text
    assert "Error en login" in doc_text


def test_redmine_issue_falta_requeridos():
    data = {
        "id": 100
        # Falta "subject", que es obligatorio
    }
    with pytest.raises(ValidationError):
        RedmineIssue(**data)
