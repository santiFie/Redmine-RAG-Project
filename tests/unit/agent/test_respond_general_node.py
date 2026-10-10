"""
tests/unit/agent/test_respond_general_node.py
============================================
Pruebas unitarias deterministas y aisladas para respond_general_node.
Evalúa la generación de respuesta general (saludos, conversación casual) sin RAG ni MCP.
"""

from unittest import mock
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

with mock.patch("src.rag.engine.RAGEngine", autospec=True):
    from src.agent.graph import respond_general_node
    from src.agent.state import State


@pytest.mark.asyncio
async def test_respond_general_exitoso_con_historial():
    """Valida la generación de respuesta casual con historial de mensajes."""
    mock_llm_response = mock.MagicMock()
    mock_llm_response.content = "¡Hola! ¿En qué puedo asistirte con Redmine hoy?"

    mock_llm = mock.AsyncMock()
    mock_llm.ainvoke.return_value = mock_llm_response

    with mock.patch("src.agent.graph.get_llm", return_value=mock_llm):
        state = {
            "messages": [HumanMessage(content="Hola, ¿qué puedes hacer?")],
        }

        res = await respond_general_node(state)

        assert res["final_answer"] == "¡Hola! ¿En qué puedo asistirte con Redmine hoy?"
        assert len(res["messages"]) == 1
        assert isinstance(res["messages"][0], AIMessage)
        assert res["messages"][0].content == res["final_answer"]

        # Validar que los mensajes enviados incluyan el SystemMessage del prompt y el mensaje humano
        invoked_msgs = mock_llm.ainvoke.call_args[0][0]
        assert isinstance(invoked_msgs[0], SystemMessage)
        assert invoked_msgs[1].content == "Hola, ¿qué puedes hacer?"


@pytest.mark.asyncio
async def test_respond_general_sin_mensajes_en_estado():
    """Caso de borde: messages está vacío o no contiene entradas."""
    mock_llm_response = mock.MagicMock()
    mock_llm_response.content = "Saludos."

    mock_llm = mock.AsyncMock()
    mock_llm.ainvoke.return_value = mock_llm_response

    with mock.patch("src.agent.graph.get_llm", return_value=mock_llm):
        state = {"messages": []}

        res = await respond_general_node(state)

        assert res["final_answer"] == "Saludos."
        invoked_msgs = mock_llm.ainvoke.call_args[0][0]
        assert len(invoked_msgs) == 1
        assert isinstance(invoked_msgs[0], SystemMessage)
