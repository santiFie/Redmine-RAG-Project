"""
tests/unit/agent/test_analyze_intent_node.py
============================================
Pruebas unitarias deterministas y aisladas para analyze_intent_node.
Evalúa la clasificación de intención del usuario (knowledge_query, incident_report, general) con mocks.
"""

from unittest import mock
import pytest

with mock.patch("src.rag.engine.RAGEngine", autospec=True):
    from src.agent.graph import analyze_intent_node
    from src.agent.state import State
from langchain_typesafe.types import ChoiceAnswer, ClassifierResponse


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expected_intent",
    ["knowledge_query", "incident_report", "general"],
)
async def test_analyze_intent_clasificaciones_validas(expected_intent):
    """Valida la correcta clasificación y asignación al estado para cada intención soportada."""
    mock_response = ClassifierResponse(
        model="openai/gpt-6-luna-decisions",
        answers={
            "intent": ChoiceAnswer(
                type="choice",
                choice=expected_intent,
                probabilities={expected_intent: 0.98},
                confidence=0.98,
            )
        },
    )
    mock_classifier = mock.AsyncMock()
    mock_classifier.ainvoke.return_value = mock_response

    with mock.patch("src.agent.graph.get_typesafe_classifier", return_value=mock_classifier):
        state = {"user_input": "Texto de prueba para intención"}
        res = await analyze_intent_node(state)

        assert res["intent"] == expected_intent
        mock_classifier.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_analyze_intent_sin_choice_intent_fallback_a_general():
    """Caso de borde: Si la respuesta del clasificador no tiene la clave 'intent', fallback determinista a 'general'."""
    mock_response = ClassifierResponse(
        model="openai/gpt-6-luna-decisions",
        answers={},
    )
    mock_classifier = mock.AsyncMock()
    mock_classifier.ainvoke.return_value = mock_response

    with mock.patch("src.agent.graph.get_typesafe_classifier", return_value=mock_classifier):
        state = {"user_input": "¿Hola, cómo estás?"}
        res = await analyze_intent_node(state)

        assert res["intent"] == "general"


@pytest.mark.asyncio
async def test_analyze_intent_user_input_vacio():
    """Caso límite: user_input es una cadena vacía."""
    mock_response = ClassifierResponse(
        model="openai/gpt-6-luna-decisions",
        answers={
            "intent": ChoiceAnswer(
                type="choice",
                choice="general",
                probabilities={"general": 1.0},
                confidence=1.0,
            )
        },
    )
    mock_classifier = mock.AsyncMock()
    mock_classifier.ainvoke.return_value = mock_response

    with mock.patch("src.agent.graph.get_typesafe_classifier", return_value=mock_classifier):
        state = {"user_input": ""}
        res = await analyze_intent_node(state)

        assert res["intent"] == "general"
        call_args = mock_classifier.ainvoke.call_args[0][0]
        assert call_args["state"] == ""
