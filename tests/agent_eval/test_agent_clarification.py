"""
tests/agent_eval/test_agent_clarification.py
============================================
Evaluación del subsistema de clarificación y suficiencia de incidentes:
  - qa_evaluator_node: evaluación de completitud, arquetipo de incidente y extracción técnica.
  - ask_clarification_node: generación de preguntas empáticas e interrupción HITL.
  - route_qa: enrutamiento condicional y prevención de bucles infinitos (escape hatch).
"""

from __future__ import annotations

from unittest import mock
import pytest
from langgraph.errors import GraphInterrupt

# Mockear RAGEngine antes de importar para evitar bloqueos por conexión a Qdrant
with mock.patch("src.rag.engine.RAGEngine", autospec=True):
    from src.agent.graph import ask_clarification_node, qa_evaluator_node, route_qa
    from src.agent.state import BugReportExtraction, State

from tests.agent_eval.fixtures import CLARIFICATION_TEST_CASES


def test_route_qa_decision_matrix():
    """Valida la matriz completa de decisiones del enrutador de QA."""
    # 1. Reporte suficiente -> debe avanzar a duplicate_and_rag_check
    state_sufficient: State = {
        "messages": [],
        "user_input": "error con datos completos",
        "is_safe_query": True,
        "intent": "incident_report",
        "bug_analysis": BugReportExtraction(
            is_sufficient=True,
            title_summary="Fallo con datos completos",
            incident_type="backend_api",
        ),
        "clarification_turns": 0,
        "final_answer": "",
    }
    assert route_qa(state_sufficient) == "duplicate_and_rag_check"

    # 2. Reporte insuficiente sin turnos previos -> debe solicitar clarificación
    state_vague: State = {
        "messages": [],
        "user_input": "no anda",
        "is_safe_query": True,
        "intent": "incident_report",
        "bug_analysis": BugReportExtraction(
            is_sufficient=False,
            missing_fields=["pasos para reproducir", "entorno"],
            title_summary="Fallo vago",
            clarification_questions=["¿Qué acción estabas realizando cuando falló?"],
        ),
        "clarification_turns": 0,
        "final_answer": "",
    }
    assert route_qa(state_vague) == "ask_clarification"

    # 3. Escape hatch: reporte insuficiente pero el usuario ya respondió 1 turno
    state_escape: State = {
        "messages": [],
        "user_input": "no anda\n[Respuesta]: No sé más detalles",
        "is_safe_query": True,
        "intent": "incident_report",
        "bug_analysis": BugReportExtraction(
            is_sufficient=False,
            missing_fields=["pasos para reproducir"],
            title_summary="Fallo sin detalles adicionales",
        ),
        "clarification_turns": 1,
        "final_answer": "",
    }
    assert route_qa(state_escape) == "duplicate_and_rag_check"


def test_clarification_fixtures_expected_routing():
    """Verifica que las fixtures de clarificación definan rutas consistentes."""
    for case in CLARIFICATION_TEST_CASES:
        if "clarification_turns" in case:
            # Caso de escape hatch
            state: State = {
                "messages": [],
                "user_input": case["user_input"],
                "is_safe_query": True,
                "intent": "incident_report",
                "bug_analysis": BugReportExtraction(
                    is_sufficient=False,
                    title_summary="Incidente con escape hatch",
                ),
                "clarification_turns": case["clarification_turns"],
                "final_answer": "",
            }
            assert route_qa(state) == case["expected_route_after_turn"]
        else:
            state = {
                "messages": [],
                "user_input": case["user_input"],
                "is_safe_query": True,
                "intent": "incident_report",
                "bug_analysis": BugReportExtraction(
                    is_sufficient=case["expected_is_sufficient"],
                    title_summary="Test Case",
                ),
                "clarification_turns": 0,
                "final_answer": "",
            }
            assert route_qa(state) == case["expected_route"]


@pytest.mark.asyncio
async def test_ask_clarification_node_triggers_interrupt():
    """Verifica que ask_clarification_node pause el flujo mediante interrupt() con los campos requeridos."""
    mock_llm = mock.AsyncMock()
    mock_llm_response = mock.MagicMock()
    mock_llm_response.content = "¿Podrías detallar qué endpoint o pantalla estabas utilizando?"
    mock_llm.ainvoke.return_value = mock_llm_response

    with (
        mock.patch("src.agent.graph.get_llm", return_value=mock_llm),
        mock.patch("src.agent.graph.interrupt") as mock_interrupt,
    ):
        mock_interrupt.return_value = "Estaba en /api/v1/checkout"

        state: State = {
            "messages": [],
            "user_input": "No funciona el pago",
            "is_safe_query": True,
            "intent": "incident_report",
            "bug_analysis": BugReportExtraction(
                is_sufficient=False,
                missing_fields=["endpoint", "código de error"],
                title_summary="Fallo en pasarela de pago",
                clarification_questions=["¿Qué endpoint estabas llamando?"],
            ),
            "clarification_turns": 0,
            "final_answer": "",
        }

        output = await ask_clarification_node(state)

        mock_interrupt.assert_called_once()
        interrupt_data = mock_interrupt.call_args[0][0]
        assert interrupt_data["action"] == "provide_clarification"
        assert "¿Podrías detallar" in interrupt_data["question"]
        assert "endpoint" in interrupt_data["missing_fields"]
        assert interrupt_data["clarification_questions"] == ["¿Qué endpoint estabas llamando?"]
        assert output["clarification_turns"] == 1
        assert "Estaba en /api/v1/checkout" in output["user_input"]


@pytest.mark.asyncio
async def test_qa_evaluator_node_populates_bug_analysis():
    """Verifica que qa_evaluator_node analice el input y almacene el resultado estructurado en el estado."""
    expected_extraction = BugReportExtraction(
        is_sufficient=True,
        title_summary="[Checkout] KeyError 'billing_address' en POST /api/v1/checkout",
        incident_type="backend_api",
        technical_details="Status 500, KeyError: 'billing_address', endpoint /api/v1/checkout",
        description_markdown="h3. Síntomas\nError 500 al enviar orden.",
        suggested_tracker="Bug",
        suggested_priority="Alta",
    )

    mock_structured_llm = mock.AsyncMock()
    mock_structured_llm.ainvoke.return_value = expected_extraction

    mock_llm = mock.MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with mock.patch("src.agent.graph.get_llm", return_value=mock_llm):
        state: State = {
            "messages": [],
            "user_input": "Al hacer POST a /api/v1/checkout da 500 KeyError 'billing_address'",
            "is_safe_query": True,
            "intent": "incident_report",
            "final_answer": "",
        }

        output = await qa_evaluator_node(state)

        assert "bug_analysis" in output
        analysis: BugReportExtraction = output["bug_analysis"]
        assert analysis.is_sufficient is True
        assert analysis.incident_type == "backend_api"
        assert analysis.suggested_priority == "Alta"
        assert "billing_address" in analysis.title_summary
