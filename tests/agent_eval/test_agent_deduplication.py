"""
tests/agent_eval/test_agent_deduplication.py
============================================
Evaluación del subsistema de triaje y deduplicación de tickets:
  - duplicate_and_rag_check_node: construcción de query enriquecida, búsqueda RAG y evaluación con LLM.
  - route_duplicates: enrutamiento condicional entre ticket existente o confirmación de nuevo ticket.
  - Control de falsos positivos y falsos negativos.
"""

from __future__ import annotations

from unittest import mock
import pytest

with mock.patch("src.rag.engine.RAGEngine", autospec=True):
    from src.agent.graph import (
        DuplicateCheckResult,
        build_graph,
        duplicate_and_rag_check_node,
        respond_existing_solution_node,
    )
    from src.agent.state import BugReportExtraction, State

from tests.agent_eval.fixtures import DEDUPLICATION_TEST_CASES


def _get_route_duplicates():
    wf = build_graph()
    branch = list(wf.branches["duplicate_and_rag_check"].values())[0]
    return branch.path


def test_route_duplicates_logic():
    """Valida la bifurcación condicional entre ticket existente y creación de nuevo ticket."""
    route_fn = _get_route_duplicates()

    state_dup: State = {
        "messages": [],
        "user_input": "",
        "is_safe_query": True,
        "intent": "incident_report",
        "is_duplicate": True,
        "duplicate_issue_id": 105,
        "final_answer": "",
    }
    assert route_fn.invoke(state_dup) == "respond_existing"

    state_new: State = {
        "messages": [],
        "user_input": "",
        "is_safe_query": True,
        "intent": "incident_report",
        "is_duplicate": False,
        "suggested_project_id": "proyecto-prueba",
        "final_answer": "",
    }
    assert route_fn.invoke(state_new) == "confirm_creation"


@pytest.mark.asyncio
async def test_duplicate_and_rag_check_node_positive_duplicate():
    """Verifica la detección exitosa de un ticket duplicado abierto."""
    case = DEDUPLICATION_TEST_CASES[0]  # Duplicado de idempotencia en pagos

    mock_rag_result = {
        "context": (
            "[Ticket #19] [Payments API] Duplicación de cargos en /api/v1/payments/charge por ausencia de encabezado Idempotency-Key\n"
            "Estado: En curso. Se han detectado transacciones duplicadas cuando los usuarios hacen doble click."
        ),
        "issue_ids": [19],
    }

    mock_rag_engine = mock.AsyncMock()
    mock_rag_engine.aquery.return_value = mock_rag_result

    eval_result = DuplicateCheckResult(
        is_duplicate=True,
        duplicate_issue_id=19,
        reasoning="El reporte describe exactamente la duplicación de cargos en /api/v1/payments/charge ya registrada en el ticket #19.",
        suggested_project_id="proyecto-prueba",
    )

    mock_structured_llm = mock.AsyncMock()
    mock_structured_llm.ainvoke.return_value = eval_result
    mock_llm = mock.MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    mock_client = mock.MagicMock()
    mock_client.list_projects.return_value = [
        {"id": 1, "identifier": "infraestructura-de-servicios", "name": "Infraestructura"},
        {"id": 2, "identifier": "proyecto-prueba", "name": "Proyecto Prueba"},
    ]

    with (
        mock.patch("src.agent.graph.get_rag_engine", return_value=mock_rag_engine),
        mock.patch("src.agent.graph.get_llm", return_value=mock_llm),
        mock.patch("src.agent.graph.RedmineClient", return_value=mock_client),
    ):
        state: State = {
            "messages": [],
            "user_input": case["user_report"],
            "is_safe_query": True,
            "intent": "incident_report",
            "bug_analysis": BugReportExtraction(
                is_sufficient=True,
                title_summary="Doble cobro en payments charge",
                incident_type="backend_api",
                technical_details="POST /api/v1/payments/charge, doble click",
            ),
            "final_answer": "",
        }

        output = await duplicate_and_rag_check_node(state)

        assert output["is_duplicate"] is True
        assert output["duplicate_issue_id"] == 19
        assert output["retrieved_issue_ids"] == [19]
        assert "Ticket #19" in output["rag_context"]


@pytest.mark.asyncio
async def test_duplicate_and_rag_check_node_false_positive_control():
    """Verifica que un reporte con endpoint similar pero causa raíz diferente NO sea marcado como duplicado."""
    case = DEDUPLICATION_TEST_CASES[3]  # Control negativo: Error en cupón de checkout

    mock_rag_result = {
        "context": (
            "[Ticket #17] [Checkout API] Error 500 por KeyError: 'billing_address' en POST /api/v1/checkout\n"
            "Estado: En curso. Falla al seleccionar misma dirección de facturación."
        ),
        "issue_ids": [17],
    }

    mock_rag_engine = mock.AsyncMock()
    mock_rag_engine.aquery.return_value = mock_rag_result

    eval_result = DuplicateCheckResult(
        is_duplicate=False,
        duplicate_issue_id=None,
        reasoning="Aunque ambos afectan /api/v1/checkout, este error ocurre por validación de caracteres en cupones (HTTP 400) y no por billing_address (HTTP 500).",
        suggested_project_id="proyecto-prueba",
    )

    mock_structured_llm = mock.AsyncMock()
    mock_structured_llm.ainvoke.return_value = eval_result
    mock_llm = mock.MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    mock_client = mock.MagicMock()
    mock_client.list_projects.return_value = [
        {"id": 1, "identifier": "infraestructura-de-servicios", "name": "Infraestructura"},
        {"id": 2, "identifier": "proyecto-prueba", "name": "Proyecto Prueba"},
    ]

    with (
        mock.patch("src.agent.graph.get_rag_engine", return_value=mock_rag_engine),
        mock.patch("src.agent.graph.get_llm", return_value=mock_llm),
        mock.patch("src.agent.graph.RedmineClient", return_value=mock_client),
    ):
        state: State = {
            "messages": [],
            "user_input": case["user_report"],
            "is_safe_query": True,
            "intent": "incident_report",
            "bug_analysis": BugReportExtraction(
                is_sufficient=True,
                title_summary="Error 400 en checkout con cupón con caracteres especiales",
                incident_type="backend_api",
                technical_details="POST /api/v1/checkout, status 400, cupones",
            ),
            "final_answer": "",
        }

        output = await duplicate_and_rag_check_node(state)

        assert output["is_duplicate"] is False
        assert output["duplicate_issue_id"] is None
        assert output["suggested_project_id"] == "proyecto-prueba"


@pytest.mark.asyncio
async def test_respond_existing_solution_node():
    """Valida que respond_existing_solution_node informe al usuario la referencia al ticket duplicado."""
    state: State = {
        "messages": [],
        "user_input": "",
        "is_safe_query": True,
        "intent": "incident_report",
        "duplicate_issue_id": 42,
        "final_answer": "",
    }
    result = await respond_existing_solution_node(state)
    assert "#42" in result["final_answer"]
    assert len(result["messages"]) == 1
