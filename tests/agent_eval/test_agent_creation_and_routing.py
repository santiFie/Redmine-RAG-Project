"""
tests/agent_eval/test_agent_creation_and_routing.py
===================================================
Evaluación de confirmación HITL, enrutamiento a proyectos y creación efectiva de tickets:
  - confirm_issue_creation_node: pausa HITL interactiva y selección o confirmación de proyecto.
  - redmine_issue_creator_node: despacho de llamada a la API REST de Redmine con formato estructurado.
  - respond_creation_summary_node: formateo del enlace directo y resumen para el usuario.
"""

from __future__ import annotations

from unittest import mock
import pytest
from langgraph.errors import GraphInterrupt

with mock.patch("src.rag.engine.RAGEngine", autospec=True):
    from src.agent.graph import (
        confirm_issue_creation_node,
        redmine_issue_creator_node,
        respond_creation_summary_node,
    )
    from src.agent.state import BugReportExtraction, State

from tests.agent_eval.fixtures import CREATION_AND_ROUTING_TEST_CASES


@pytest.mark.asyncio
async def test_confirm_issue_creation_node_triggers_interrupt():
    """Verifica que confirm_issue_creation_node pause el flujo para confirmación del proyecto."""
    state: State = {
        "messages": [],
        "user_input": "",
        "is_safe_query": True,
        "intent": "incident_report",
        "suggested_project_id": "infraestructura-de-servicios",
        "available_projects": [
            {"id": 1, "identifier": "infraestructura-de-servicios", "name": "Infraestructura"},
            {"id": 2, "identifier": "proyecto-prueba", "name": "Proyecto Prueba"},
        ],
        "bug_analysis": BugReportExtraction(
            is_sufficient=True,
            title_summary="Fallo de red en pod",
        ),
        "final_answer": "",
    }

    with mock.patch("src.agent.graph.interrupt") as mock_interrupt:
        mock_interrupt.return_value = "infraestructura-de-servicios"

        output = await confirm_issue_creation_node(state)

        mock_interrupt.assert_called_once()
        interrupt_data = mock_interrupt.call_args[0][0]
        assert interrupt_data["action"] == "confirm_issue_creation"
        assert interrupt_data["suggested_project"] == "infraestructura-de-servicios"
        assert len(interrupt_data["available_projects"]) == 2
        assert output["target_project_id"] == "infraestructura-de-servicios"


@pytest.mark.asyncio
async def test_redmine_issue_creator_node_structured_ticket(monkeypatch):
    """Verifica que el creador envíe los metadatos y cuerpo estructurado en Redmine."""
    monkeypatch.setenv("REDMINE_URL", "http://redmine.corp.internal:3000")

    mock_client = mock.MagicMock()
    mock_client.create_issue.return_value = {"id": 8812}

    with mock.patch("src.agent.graph.RedmineClient", return_value=mock_client):
        custom_markdown = (
            "h3. Síntomas\nConexiones rechazadas a las 04:00 UTC.\n\n"
            "h3. Detalles Técnicos\nFATAL: sorry, too many clients already\n\n"
            "h3. Impacto\nCrítico en producción."
        )

        state: State = {
            "messages": [],
            "user_input": "",
            "is_safe_query": True,
            "intent": "incident_report",
            "target_project_id": "infraestructura-de-servicios",
            "bug_analysis": BugReportExtraction(
                is_sufficient=True,
                title_summary="[PostgreSQL] Saturación de conexiones en pgBouncer",
                incident_type="infra_outage",
                description_markdown=custom_markdown,
            ),
            "final_answer": "",
        }

        result = await redmine_issue_creator_node(state)

        assert result["created_issue_id"] == 8812
        assert result["created_issue_url"] == "http://redmine.corp.internal:3000/issues/8812"

        mock_client.create_issue.assert_called_once_with(
            project_id="infraestructura-de-servicios",
            subject="[PostgreSQL] Saturación de conexiones en pgBouncer",
            description=custom_markdown,
        )


@pytest.mark.asyncio
async def test_respond_creation_summary_node():
    """Verifica que la respuesta final contenga la URL del issue recién creado."""
    state: State = {
        "messages": [],
        "user_input": "",
        "is_safe_query": True,
        "intent": "incident_report",
        "created_issue_url": "http://redmine.corp.internal:3000/issues/8812",
        "final_answer": "",
    }
    result = await respond_creation_summary_node(state)
    assert "http://redmine.corp.internal:3000/issues/8812" in result["final_answer"]
    assert len(result["messages"]) == 1


def test_creation_fixtures_project_mapping():
    """Verifica la consistencia de los casos de prueba de enrutamiento por tipología."""
    for case in CREATION_AND_ROUTING_TEST_CASES:
        assert case["expected_project_id"] in [
            "infraestructura-de-servicios",
            "proyecto-prueba",
        ]
        assert case["expected_tracker"] in ["Bug", "Feature", "Soporte"]
