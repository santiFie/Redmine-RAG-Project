"""
tests/unit/agent/test_redmine_issue_creator_node.py
===================================================
Pruebas unitarias deterministas y aisladas para redmine_issue_creator_node.
Evalúa la invocación mockeada a RedmineClient.create_issue, generación del payload y formateo de URLs.
"""

from unittest import mock
import pytest

with mock.patch("src.rag.engine.RAGEngine", autospec=True):
    from src.agent.graph import redmine_issue_creator_node
    from src.agent.state import BugReportExtraction, State


@pytest.mark.asyncio
async def test_redmine_issue_creator_con_description_markdown(monkeypatch):
    """Valida la creación del issue usando la descripción formateada en Markdown/Textile del bug_analysis."""
    monkeypatch.setenv("REDMINE_URL", "http://redmine.corp.local:3000")

    mock_client = mock.MagicMock()
    mock_client.create_issue.return_value = {"id": 12345}

    with mock.patch("src.agent.graph.RedmineClient", return_value=mock_client):
        custom_body = (
            "h3. Incidente de Producción\nFallo crítico en facturación.\nh3. Impacto\nAlto"
        )
        state = {
            "target_project_id": "billing",
            "bug_analysis": BugReportExtraction(
                is_sufficient=True,
                title_summary="Fallo en facturación",
                description_markdown=custom_body,
            ),
        }

        res = await redmine_issue_creator_node(state)

        assert res["created_issue_id"] == 12345
        assert res["created_issue_url"] == "http://redmine.corp.local:3000/issues/12345"
        mock_client.create_issue.assert_called_once_with(
            project_id="billing",
            subject="Fallo en facturación",
            description=custom_body,
        )


@pytest.mark.asyncio
async def test_redmine_issue_creator_fallback_pasos_y_entorno(monkeypatch):
    """Valida la reconstrucción del cuerpo a partir de pasos y entorno si description_markdown está vacía."""
    monkeypatch.setenv("REDMINE_URL", "http://redmine.corp.local:3000")

    mock_client = mock.MagicMock()
    mock_client.create_issue.return_value = {"id": 6789}

    with mock.patch("src.agent.graph.RedmineClient", return_value=mock_client):
        state = {
            "target_project_id": "core",
            "bug_analysis": BugReportExtraction(
                is_sufficient=True,
                title_summary="Error 500 en login",
                description_markdown="",
                reproduction_steps="1. Abrir login\n2. Enviar form",
                environment_info="Chrome v122 en Linux",
            ),
        }

        res = await redmine_issue_creator_node(state)

        assert res["created_issue_id"] == 6789
        expected_desc = "h3. Pasos para Reproducir\n1. Abrir login\n2. Enviar form\n\nh3. Entorno\nChrome v122 en Linux"
        mock_client.create_issue.assert_called_once_with(
            project_id="core",
            subject="Error 500 en login",
            description=expected_desc,
        )


@pytest.mark.asyncio
async def test_redmine_issue_creator_sin_bug_analysis():
    """Caso de borde: Si bug_analysis es None, retorna diccionario vacío sin llamar al cliente."""
    state = {"bug_analysis": None}
    res = await redmine_issue_creator_node(state)
    assert res == {}


@pytest.mark.asyncio
async def test_redmine_issue_creator_fallback_a_suggested_y_default_project(monkeypatch):
    """Valida el orden de precedencia para project_id: target -> suggested -> default_env."""
    monkeypatch.setenv("REDMINE_URL", "http://redmine.corp.local:3000")
    monkeypatch.setenv("REDMINE_DEFAULT_PROJECT", "env-default-proj")

    mock_client = mock.MagicMock()
    mock_client.create_issue.return_value = {"id": 111}

    # 1. Sin target_project_id pero con suggested_project_id
    with mock.patch("src.agent.graph.RedmineClient", return_value=mock_client):
        state = {
            "suggested_project_id": "suggested-proj",
            "bug_analysis": BugReportExtraction(
                is_sufficient=True,
                title_summary="Bug de prueba",
                description_markdown="desc",
            ),
        }
        await redmine_issue_creator_node(state)
        assert mock_client.create_issue.call_args[1]["project_id"] == "suggested-proj"

    # 2. Sin target ni suggested -> usa REDMINE_DEFAULT_PROJECT
    mock_client.reset_mock()
    with mock.patch("src.agent.graph.RedmineClient", return_value=mock_client):
        state_fallback = {
            "bug_analysis": BugReportExtraction(
                is_sufficient=True,
                title_summary="Bug sin proyecto",
                description_markdown="desc",
            ),
        }
        await redmine_issue_creator_node(state_fallback)
        assert mock_client.create_issue.call_args[1]["project_id"] == "env-default-proj"
