"""
tests/unit/agent/test_confirm_issue_creation_node.py
====================================================
Pruebas unitarias deterministas y aisladas para confirm_issue_creation_node.
Evalúa la pausa interactiva HITL (interrupt()) y el procesamiento de respuestas de confirmación o sobreescritura.
"""

from unittest import mock
import pytest
from langchain_core.messages import AIMessage, HumanMessage

with mock.patch("src.rag.engine.RAGEngine", autospec=True):
    from src.agent.graph import confirm_issue_creation_node
    from src.agent.state import BugReportExtraction, State


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "affirmative_answer", ["si", "sí", "yes", "confirmar", "ok", "confirmo", "  Sí  "]
)
async def test_confirm_issue_creation_respuesta_afirmativa_conserva_proyecto(affirmative_answer):
    """Valida que respuestas afirmativas confirmen el proyecto sugerido por defecto."""
    with mock.patch("src.agent.graph.interrupt", return_value=affirmative_answer) as mock_interrupt:
        state = {
            "suggested_project_id": "proyecto-sugerido",
            "available_projects": [
                {"id": 1, "identifier": "proyecto-sugerido", "name": "Proyecto Sugerido"}
            ],
            "bug_analysis": BugReportExtraction(
                is_sufficient=True,
                title_summary="Fallo en pasarela",
            ),
        }

        res = await confirm_issue_creation_node(state)

        mock_interrupt.assert_called_once()
        assert res["target_project_id"] == "proyecto-sugerido"
        assert len(res["messages"]) == 2
        assert isinstance(res["messages"][0], AIMessage)
        assert isinstance(res["messages"][1], HumanMessage)


@pytest.mark.asyncio
async def test_confirm_issue_creation_sobreescritura_string():
    """Valida que si el usuario especifica un ID/identifier alternativo como texto, se adopte como target_project_id."""
    with mock.patch("src.agent.graph.interrupt", return_value="pagos-v2"):
        state = {
            "suggested_project_id": "proyecto-anterior",
            "available_projects": [],
            "bug_analysis": BugReportExtraction(
                is_sufficient=True,
                title_summary="Error en checkout",
            ),
        }

        res = await confirm_issue_creation_node(state)

        assert res["target_project_id"] == "pagos-v2"
        assert res["messages"][1].content == "pagos-v2"


@pytest.mark.asyncio
async def test_confirm_issue_creation_sobreescritura_dict():
    """Valida la recepción de respuestas estructuradas desde la interfaz de usuario en formato diccionario."""
    with mock.patch("src.agent.graph.interrupt", return_value={"project_id": "infra-cloud"}):
        state = {
            "suggested_project_id": "core-backend",
            "available_projects": [],
            "bug_analysis": BugReportExtraction(
                is_sufficient=True,
                title_summary="Caída de nodo",
            ),
        }

        res = await confirm_issue_creation_node(state)

        assert res["target_project_id"] == "infra-cloud"


@pytest.mark.asyncio
async def test_confirm_issue_creation_sin_bug_analysis_ni_proyecto(monkeypatch):
    """Caso límite: bug_analysis y suggested_project_id no provistos en el estado."""
    monkeypatch.setenv("REDMINE_DEFAULT_PROJECT", "fallback-project")

    with mock.patch("src.agent.graph.interrupt", return_value="sí") as mock_interrupt:
        state = {}

        res = await confirm_issue_creation_node(state)

        assert res["target_project_id"] == "fallback-project"
        interrupt_call = mock_interrupt.call_args[0][0]
        assert "Incidente" in interrupt_call["question"]
        assert interrupt_call["suggested_project"] == "fallback-project"
