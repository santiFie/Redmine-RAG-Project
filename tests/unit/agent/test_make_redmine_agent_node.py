"""
tests/unit/agent/test_make_redmine_agent_node.py
================================================
Pruebas unitarias deterministas y aisladas para make_redmine_agent_node y get_redmine_tools.
Evalúa la construcción del subgrafo de Redmine ReAct, la interacción mockeada con el cliente MCP y el fallback de respuestas.
"""

from contextlib import asynccontextmanager
from unittest import mock
import pytest
from langchain_core.messages import AIMessage, HumanMessage

with mock.patch("src.rag.engine.RAGEngine", autospec=True):
    from src.agent.graph import get_redmine_tools, make_redmine_agent_node
    from src.agent.state import State


@pytest.mark.asyncio
async def test_make_redmine_agent_node_ejecucion_exitosa():
    """Valida la ejecución del sub-grafo MCP de Redmine retornando una respuesta de IA válida."""
    mock_llm = mock.MagicMock()
    mock_llm_with_tools = mock.AsyncMock()
    # Retorna un AIMessage sin tool_calls para que el subgrafo termine directamente en END
    mock_llm_with_tools.ainvoke.return_value = AIMessage(
        content="El ticket #204 ha sido actualizado exitosamente con la nota."
    )
    mock_llm.bind_tools.return_value = mock_llm_with_tools

    mock_client = mock.MagicMock()
    mock_session = mock.AsyncMock()

    @asynccontextmanager
    async def fake_session(server_name):
        yield mock_session

    mock_client.session.side_effect = fake_session

    with (
        mock.patch("src.agent.graph.get_llm", return_value=mock_llm),
        mock.patch("src.agent.graph.MultiServerMCPClient", return_value=mock_client),
        mock.patch("src.agent.graph.load_mcp_tools", new=mock.AsyncMock(return_value=[])),
    ):
        node = make_redmine_agent_node()

        state: State = {
            "messages": [HumanMessage(content="Actualiza el ticket #204")],
            "user_input": "Actualiza el ticket #204",
            "is_safe_query": True,
            "intent": "incident_report",
            "final_answer": "",
        }

        res = await node(state)

        assert (
            res["redmine_result"] == "El ticket #204 ha sido actualizado exitosamente con la nota."
        )
        assert res["final_answer"] == res["redmine_result"]
        assert len(res["messages"]) == 1
        assert isinstance(res["messages"][0], AIMessage)
        assert res["messages"][0].content == res["final_answer"]


@pytest.mark.asyncio
async def test_make_redmine_agent_node_sin_respuesta_de_ia():
    """Caso de borde: El modelo no produce contenido en el mensaje de salida."""
    mock_llm = mock.MagicMock()
    mock_llm_with_tools = mock.AsyncMock()
    mock_llm_with_tools.ainvoke.return_value = AIMessage(content="")
    mock_llm.bind_tools.return_value = mock_llm_with_tools

    mock_client = mock.MagicMock()
    mock_session = mock.AsyncMock()

    @asynccontextmanager
    async def fake_session(server_name):
        yield mock_session

    mock_client.session.side_effect = fake_session

    with (
        mock.patch("src.agent.graph.get_llm", return_value=mock_llm),
        mock.patch("src.agent.graph.MultiServerMCPClient", return_value=mock_client),
        mock.patch("src.agent.graph.load_mcp_tools", new=mock.AsyncMock(return_value=[])),
    ):
        node = make_redmine_agent_node()

        state = {
            "messages": [HumanMessage(content="Mensaje sin respuesta")],
        }

        res = await node(state)

        assert res["redmine_result"] == "Sin respuesta del agente de Redmine."
        assert res["final_answer"] == "Sin respuesta del agente de Redmine."


@pytest.mark.asyncio
async def test_get_redmine_tools_filtrado_por_nombre():
    """Valida la carga y filtrado de herramientas MCP por nombre permitido."""
    tool1 = mock.MagicMock()
    tool1.name = "redmine_get_issue"
    tool2 = mock.MagicMock()
    tool2.name = "redmine_delete_issue"

    mock_client = mock.MagicMock()
    mock_client.get_tools = mock.AsyncMock(return_value=[tool1, tool2])

    with mock.patch("src.agent.graph.MultiServerMCPClient", return_value=mock_client):
        # 1. Sin filtro
        all_tools = await get_redmine_tools()
        assert len(all_tools) == 2

        # 2. Con filtro de nombres permitidos
        filtered_tools = await get_redmine_tools(allowed_names={"redmine_get_issue"})
        assert len(filtered_tools) == 1
        assert filtered_tools[0].name == "redmine_get_issue"
