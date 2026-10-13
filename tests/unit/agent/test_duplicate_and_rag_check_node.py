"""
tests/unit/agent/test_duplicate_and_rag_check_node.py
=====================================================
Pruebas unitarias deterministas y aisladas para duplicate_and_rag_check_node.
Evalúa la deduplicación semántica con RAG, la consulta de proyectos de Redmine y la sugerencia de proyecto.
"""

from unittest import mock
import pytest

with mock.patch("src.rag.engine.RAGEngine", autospec=True):
    from src.agent.graph import DuplicateCheckResult, duplicate_and_rag_check_node
    from src.agent.state import BugReportExtraction, State


@pytest.mark.asyncio
async def test_duplicate_check_detecta_duplicado():
    """Valida la detección exitosa de un ticket duplicado mediante RAG y el modelo estructurado."""
    mock_rag_engine = mock.AsyncMock()
    mock_rag_engine.aquery.return_value = {
        "context": "Ticket #42: Error 500 al guardar factura. En progreso.",
        "issue_ids": [42],
    }

    mock_client = mock.MagicMock()
    mock_client.list_projects.return_value = [
        {"id": 1, "identifier": "invoicing", "name": "Facturación"},
    ]

    eval_result = DuplicateCheckResult(
        is_duplicate=True,
        duplicate_issue_id=42,
        reasoning="El error 500 al guardar factura coincide exactamente con el ticket #42.",
        suggested_project_id="invoicing",
    )
    mock_structured_llm = mock.AsyncMock()
    mock_structured_llm.ainvoke.return_value = eval_result
    mock_llm = mock.MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with (
        mock.patch("src.agent.graph.get_rag_engine", return_value=mock_rag_engine),
        mock.patch("src.agent.graph.RedmineClient", return_value=mock_client),
        mock.patch("src.agent.graph.get_llm", return_value=mock_llm),
    ):
        state = {
            "bug_analysis": BugReportExtraction(
                is_sufficient=True,
                title_summary="Fallo 500 al emitir factura",
                incident_type="backend_api",
                technical_details="POST /invoices retorna 500",
            )
        }
        res = await duplicate_and_rag_check_node(state)

        assert res["is_duplicate"] is True
        assert res["duplicate_issue_id"] == 42
        assert res["suggested_project_id"] == "invoicing"
        assert res["retrieved_issue_ids"] == [42]


@pytest.mark.asyncio
async def test_duplicate_check_sin_duplicado_con_proyecto_sugerido():
    """Valida el escenario donde no hay duplicados y se propone el proyecto adecuado."""
    mock_rag_engine = mock.AsyncMock()
    mock_rag_engine.aquery.return_value = {"context": "", "issue_ids": []}

    mock_client = mock.MagicMock()
    mock_client.list_projects.return_value = [
        {"id": 10, "identifier": "auth-service", "name": "Autenticación"},
        {"id": 20, "identifier": "billing", "name": "Facturación"},
    ]

    eval_result = DuplicateCheckResult(
        is_duplicate=False,
        duplicate_issue_id=None,
        reasoning="Nuevo bug reportado para el servicio de autenticación.",
        suggested_project_id="auth-service",
    )
    mock_structured_llm = mock.AsyncMock()
    mock_structured_llm.ainvoke.return_value = eval_result
    mock_llm = mock.MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with (
        mock.patch("src.agent.graph.get_rag_engine", return_value=mock_rag_engine),
        mock.patch("src.agent.graph.RedmineClient", return_value=mock_client),
        mock.patch("src.agent.graph.get_llm", return_value=mock_llm),
    ):
        state = {
            "bug_analysis": BugReportExtraction(
                is_sufficient=True,
                title_summary="Error en validación de token JWT",
            )
        }
        res = await duplicate_and_rag_check_node(state)

        assert res["is_duplicate"] is False
        assert res["duplicate_issue_id"] is None
        assert res["suggested_project_id"] == "auth-service"


@pytest.mark.asyncio
async def test_duplicate_check_sin_bug_analysis():
    """Caso de borde: Si bug_analysis es None o ausente, retorna diccionario vacío sin realizar llamadas."""
    state = {"bug_analysis": None}
    res = await duplicate_and_rag_check_node(state)
    assert res == {}


@pytest.mark.asyncio
async def test_duplicate_check_tolerancia_error_en_rag():
    """Caso de borde: Si el motor RAG lanza excepción, continúa sin romper el flujo."""
    mock_rag_engine = mock.AsyncMock()
    mock_rag_engine.aquery.side_effect = ConnectionError("Qdrant no responde")

    mock_client = mock.MagicMock()
    mock_client.list_projects.return_value = [{"id": 1, "identifier": "proj-1"}]

    eval_result = DuplicateCheckResult(
        is_duplicate=False,
        duplicate_issue_id=None,
        reasoning="Fallo RAG manejado.",
        suggested_project_id="proj-1",
    )
    mock_structured_llm = mock.AsyncMock()
    mock_structured_llm.ainvoke.return_value = eval_result
    mock_llm = mock.MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with (
        mock.patch("src.agent.graph.get_rag_engine", return_value=mock_rag_engine),
        mock.patch("src.agent.graph.RedmineClient", return_value=mock_client),
        mock.patch("src.agent.graph.get_llm", return_value=mock_llm),
    ):
        state = {
            "bug_analysis": BugReportExtraction(
                is_sufficient=True,
                title_summary="Bug con fallo de red",
            )
        }
        res = await duplicate_and_rag_check_node(state)

        assert res["is_duplicate"] is False
        assert res["rag_context"] == ""
        assert res["retrieved_issue_ids"] == []


@pytest.mark.asyncio
async def test_duplicate_check_tolerancia_error_redmine_client(monkeypatch):
    """Caso de borde: Si list_projects de RedmineClient falla, usa el proyecto default de entorno."""
    monkeypatch.setenv("REDMINE_DEFAULT_PROJECT", "default-fallback-proj")

    mock_rag_engine = mock.AsyncMock()
    mock_rag_engine.aquery.return_value = {"context": "", "issue_ids": []}

    mock_client = mock.MagicMock()
    mock_client.list_projects.side_effect = RuntimeError("Error en API de Redmine")

    eval_result = DuplicateCheckResult(
        is_duplicate=False,
        duplicate_issue_id=None,
        reasoning="Fallback a proyecto por defecto.",
        suggested_project_id=None,  # LLM tampoco sugiere
    )
    mock_structured_llm = mock.AsyncMock()
    mock_structured_llm.ainvoke.return_value = eval_result
    mock_llm = mock.MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with (
        mock.patch("src.agent.graph.get_rag_engine", return_value=mock_rag_engine),
        mock.patch("src.agent.graph.RedmineClient", return_value=mock_client),
        mock.patch("src.agent.graph.get_llm", return_value=mock_llm),
    ):
        state = {
            "bug_analysis": BugReportExtraction(
                is_sufficient=True,
                title_summary="Fallo de prueba",
            )
        }
        res = await duplicate_and_rag_check_node(state)

        assert res["available_projects"] == []
        assert res["suggested_project_id"] == "default-fallback-proj"


@pytest.mark.asyncio
async def test_duplicate_check_fallback_primer_proyecto_disponible():
    """Caso de borde: Si el LLM retorna suggested_project_id=None, asigna el primer proyecto disponible."""
    mock_rag_engine = mock.AsyncMock()
    mock_rag_engine.aquery.return_value = {"context": "", "issue_ids": []}

    mock_client = mock.MagicMock()
    mock_client.list_projects.return_value = [
        {"id": 99, "identifier": "primer-proyecto-ident", "name": "Primer Proyecto"},
        {"id": 100, "identifier": "segundo-proyecto", "name": "Segundo Proyecto"},
    ]

    eval_result = DuplicateCheckResult(
        is_duplicate=False,
        duplicate_issue_id=None,
        reasoning="Sin sugerencia explícita",
        suggested_project_id=None,
    )
    mock_structured_llm = mock.AsyncMock()
    mock_structured_llm.ainvoke.return_value = eval_result
    mock_llm = mock.MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with (
        mock.patch("src.agent.graph.get_rag_engine", return_value=mock_rag_engine),
        mock.patch("src.agent.graph.RedmineClient", return_value=mock_client),
        mock.patch("src.agent.graph.get_llm", return_value=mock_llm),
    ):
        state = {
            "bug_analysis": BugReportExtraction(
                is_sufficient=True,
                title_summary="Incidente sin proyecto sugerido",
            )
        }
        res = await duplicate_and_rag_check_node(state)

        assert res["suggested_project_id"] == "primer-proyecto-ident"
