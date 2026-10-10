"""
tests/unit/agent/test_ask_clarification_node.py
===============================================
Pruebas unitarias deterministas y aisladas para ask_clarification_node.
Evalúa la construcción de preguntas aclaratorias, la pausa HITL con interrupt() y el contador de turnos.
"""

from unittest import mock
import pytest
from langchain_core.messages import AIMessage, HumanMessage

with mock.patch("src.rag.engine.RAGEngine", autospec=True):
    from src.agent.graph import ask_clarification_node
    from src.agent.state import BugReportExtraction, State


@pytest.mark.asyncio
async def test_ask_clarification_con_clarification_questions():
    """Valida la generación de preguntas de clarificación basadas en clarification_questions y el interrupt."""
    mock_llm_response = mock.MagicMock()
    mock_llm_response.content = "¿Podrías indicarnos en qué navegador y versión ocurre el fallo?"

    mock_llm = mock.AsyncMock()
    mock_llm.ainvoke.return_value = mock_llm_response

    with (
        mock.patch("src.agent.graph.get_llm", return_value=mock_llm),
        mock.patch("src.agent.graph.interrupt") as mock_interrupt,
    ):
        mock_interrupt.return_value = "Ocurre en Firefox 124 sobre Ubuntu 22.04"

        state: State = {
            "messages": [],
            "user_input": "El botón de exportar PDF no responde",
            "is_safe_query": True,
            "intent": "incident_report",
            "clarification_turns": 0,
            "bug_analysis": BugReportExtraction(
                is_sufficient=False,
                title_summary="Fallo en botón de exportar PDF",
                missing_fields=["navegador", "versión"],
                clarification_questions=["¿Qué navegador y versión utilizas?"],
            ),
            "final_answer": "",
        }

        res = await ask_clarification_node(state)

        # Validar la invocación del interrupt HITL
        mock_interrupt.assert_called_once()
        interrupt_payload = mock_interrupt.call_args[0][0]
        assert interrupt_payload["action"] == "provide_clarification"
        assert (
            interrupt_payload["question"]
            == "¿Podrías indicarnos en qué navegador y versión ocurre el fallo?"
        )
        assert interrupt_payload["clarification_questions"] == [
            "¿Qué navegador y versión utilizas?"
        ]

        # Validar actualización del estado
        assert res["clarification_turns"] == 1
        assert "Firefox 124" in res["user_input"]
        assert len(res["messages"]) == 2
        assert isinstance(res["messages"][0], AIMessage)
        assert isinstance(res["messages"][1], HumanMessage)
        assert res["messages"][1].content == "Ocurre en Firefox 124 sobre Ubuntu 22.04"


@pytest.mark.asyncio
async def test_ask_clarification_con_missing_fields_solamente():
    """Valida que si no hay clarification_questions, formatee missing_fields en el prompt."""
    mock_llm_response = mock.MagicMock()
    mock_llm_response.content = "Por favor proporciona el endpoint y código de estado."

    mock_llm = mock.AsyncMock()
    mock_llm.ainvoke.return_value = mock_llm_response

    with (
        mock.patch("src.agent.graph.get_llm", return_value=mock_llm),
        mock.patch("src.agent.graph.interrupt", return_value="Endpoint /api/v2/sync"),
    ):
        state = {
            "user_input": "Falla la sincronización",
            "clarification_turns": 0,
            "bug_analysis": BugReportExtraction(
                is_sufficient=False,
                title_summary="Falla sincronización",
                missing_fields=["endpoint", "código HTTP"],
                clarification_questions=[],
            ),
        }

        res = await ask_clarification_node(state)

        assert res["clarification_turns"] == 1
        assert "Endpoint /api/v2/sync" in res["user_input"]


@pytest.mark.asyncio
async def test_ask_clarification_sin_bug_analysis():
    """Caso límite: bug_analysis no está presente en el estado (None)."""
    mock_llm_response = mock.MagicMock()
    mock_llm_response.content = "¿Podrías dar más detalles del problema?"

    mock_llm = mock.AsyncMock()
    mock_llm.ainvoke.return_value = mock_llm_response

    with (
        mock.patch("src.agent.graph.get_llm", return_value=mock_llm),
        mock.patch("src.agent.graph.interrupt", return_value="Respuesta de usuario"),
    ):
        state = {
            "user_input": "Error genérico",
            "clarification_turns": 0,
            "bug_analysis": None,
        }

        res = await ask_clarification_node(state)

        assert res["clarification_turns"] == 1
        assert "Respuesta de usuario" in res["user_input"]


@pytest.mark.asyncio
async def test_ask_clarification_incremento_de_turnos():
    """Valida el incremento secuencial del contador clarification_turns."""
    mock_llm_response = mock.MagicMock()
    mock_llm_response.content = "Segunda pregunta aclaratoria"

    mock_llm = mock.AsyncMock()
    mock_llm.ainvoke.return_value = mock_llm_response

    with (
        mock.patch("src.agent.graph.get_llm", return_value=mock_llm),
        mock.patch("src.agent.graph.interrupt", return_value="Datos adicionales"),
    ):
        state = {
            "user_input": "Reintento de clarificación",
            "clarification_turns": 2,
        }

        res = await ask_clarification_node(state)

        assert res["clarification_turns"] == 3
