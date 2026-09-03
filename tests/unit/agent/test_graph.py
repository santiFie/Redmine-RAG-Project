"""
tests/unit/agent/test_graph.py
==============================
Pruebas unitarias para el enrutamiento y la validación de esquemas (Pydantic) del grafo.
No incluye invocaciones reales ni mockeadas a los LLMs, enfocado en la lógica del DAG.
"""

from unittest import mock

import pytest
from pydantic import ValidationError

# Mockear RAGEngine ANTES de importar src.agent.graph para evitar
# intentos de conexión a Qdrant (I/O bloqueante) durante la recolección de tests.
with mock.patch("src.rag.engine.RAGEngine", autospec=True):
    from src.agent.graph import (
        DuplicateCheckResult,
        IntentClassification,
        SafeQueryClassification,
        build_graph,
        redmine_issue_creator_node,
        route_after_analyze,
    )
from src.agent.state import BugReportExtraction


def test_safe_query_classification_schema_valido():
    """Valida el esquema Pydantic para clasificación segura."""
    data = {"reasoning": "La consulta solo pide buscar en RAG", "is_safe_query": True}
    obj = SafeQueryClassification(**data)
    assert obj.is_safe_query is True
    assert "buscar en RAG" in obj.reasoning


def test_safe_query_classification_schema_invalido():
    """Falla si faltan campos obligatorios."""
    with pytest.raises(ValidationError):
        SafeQueryClassification(is_safe_query=True)  # falta reasoning


def test_intent_classification_schema_valido():
    """Valida el esquema Pydantic para intenciones de redmine/rag/general."""
    data = {"reasoning": "Quiere listar issues", "intent": "redmine_mcp"}
    obj = IntentClassification(**data)
    assert obj.intent == "redmine_mcp"


def test_intent_classification_schema_invalido():
    """Falla si la intención no está en los valores permitidos (Literal)."""
    with pytest.raises(ValidationError):
        IntentClassification(reasoning="algo", intent="intencion_inventada")


def test_route_after_analyze():
    """Verifica que el enrutador condicional principal funcione correctamente."""

    # Caso 1: RAG
    estado_rag = {"intent": "rag_query"}
    assert route_after_analyze(estado_rag) == "rag_query"

    # Caso 2: Redmine MCP
    estado_mcp = {"intent": "redmine_mcp"}
    assert route_after_analyze(estado_mcp) == "redmine_agent"

    # Caso 3: General
    estado_general = {"intent": "general"}
    assert route_after_analyze(estado_general) == "respond_general"

    # Caso 4: Intención desconocida o no seteada (fallback a general)
    estado_vacio = {}
    assert route_after_analyze(estado_vacio) == "respond_general"


def test_build_graph_compiles_successfully():
    """
    Comprueba que el builder del grafo no arroje errores de topología (DAG)
    y devuelva un grafo compilado listo para invocarse.
    """
    workflow = build_graph()
    graph = workflow.compile()

    # Aseguramos que se creó un CompiledStateGraph y tiene nodos iniciales
    assert graph is not None
    assert hasattr(graph, "invoke")


def test_duplicate_check_result_schema():
    """Valida que DuplicateCheckResult soporte suggested_project_id."""
    res = DuplicateCheckResult(
        is_duplicate=False,
        reasoning="No hay tickets parecidos, pertenece al CRM",
        suggested_project_id="sales-crm",
    )
    assert res.is_duplicate is False
    assert res.suggested_project_id == "sales-crm"
    assert res.duplicate_issue_id is None


@pytest.mark.asyncio
async def test_redmine_issue_creator_node_url_format(monkeypatch):
    """Verifica que el creador use RedmineClient y devuelva {REDMINE_URL}/issues/{id}."""
    monkeypatch.setenv("REDMINE_URL", "http://redmine.corp.local:3000")

    mock_client = mock.MagicMock()
    mock_client.create_issue.return_value = {"id": 9876}

    with mock.patch("src.agent.graph.RedmineClient", return_value=mock_client):
        state = {
            "bug_analysis": BugReportExtraction(
                is_sufficient=True,
                title_summary="Falla en autenticación OAuth",
                reproduction_steps="1. Click login\n2. Error 500",
                environment_info="Chrome v120 / Linux",
            ),
            "target_project_id": "auth-service",
        }
        res = await redmine_issue_creator_node(state)

        assert res["created_issue_id"] == 9876
        assert res["created_issue_url"] == "http://redmine.corp.local:3000/issues/9876"
        mock_client.create_issue.assert_called_once_with(
            project_id="auth-service",
            subject="Falla en autenticación OAuth",
            description="h3. Pasos para Reproducir\n1. Click login\n2. Error 500\n\nh3. Entorno\nChrome v120 / Linux",
        )
