"""
tests/unit/agent/test_respond_knowledge_node.py
===============================================
Pruebas unitarias deterministas y aisladas para respond_knowledge_node.
Evalúa la generación de respuesta final a partir del contexto RAG y enlaces de referencia.
"""

from unittest import mock
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

with mock.patch("src.rag.engine.RAGEngine", autospec=True):
    from src.agent.graph import respond_knowledge_node
    from src.agent.state import State


@pytest.mark.asyncio
async def test_respond_knowledge_exitoso_con_referencias(monkeypatch):
    """Valida la generación de respuesta exitosa e inclusión de instrucciones para links de tickets."""
    monkeypatch.setenv("REDMINE_HOST_URL", "http://redmine.corp:3000")

    mock_llm_response = mock.MagicMock()
    mock_llm_response.content = "Para configurar LDAP debes ir a Administración > Autenticación."

    mock_llm = mock.AsyncMock()
    mock_llm.ainvoke.return_value = mock_llm_response

    with mock.patch("src.agent.graph.get_llm", return_value=mock_llm):
        state: State = {
            "messages": [HumanMessage(content="¿Cómo configuro LDAP?")],
            "user_input": "¿Cómo configuro LDAP?",
            "is_safe_query": True,
            "intent": "knowledge_query",
            "rag_context": "Documentación oficial sobre LDAP en Redmine.",
            "retrieved_issue_ids": [101, 102],
            "final_answer": "",
        }

        res = await respond_knowledge_node(state)

        assert (
            res["final_answer"] == "Para configurar LDAP debes ir a Administración > Autenticación."
        )
        assert len(res["messages"]) == 1
        assert isinstance(res["messages"][0], AIMessage)
        assert res["messages"][0].content == res["final_answer"]

        # Validar que los mensajes pasados al LLM contengan la sección de tickets
        invoked_messages = mock_llm.ainvoke.call_args[0][0]
        sys_ref_msg = next(
            (
                m
                for m in invoked_messages
                if isinstance(m, SystemMessage) and "Tickets de Referencia" in m.content
            ),
            None,
        )
        assert sys_ref_msg is not None
        assert "http://redmine.corp:3000/issues/101" in sys_ref_msg.content
        assert "http://redmine.corp:3000/issues/102" in sys_ref_msg.content


@pytest.mark.asyncio
async def test_respond_knowledge_exitoso_sin_referencias():
    """Valida la generación de respuesta cuando retrieved_issue_ids está vacío."""
    mock_llm_response = mock.MagicMock()
    mock_llm_response.content = "Respuesta sin tickets de referencia."

    mock_llm = mock.AsyncMock()
    mock_llm.ainvoke.return_value = mock_llm_response

    with mock.patch("src.agent.graph.get_llm", return_value=mock_llm):
        state = {
            "messages": [HumanMessage(content="Consulta general de conocimiento")],
            "rag_context": "Contexto relevante sin IDs de ticket.",
            "retrieved_issue_ids": [],
        }

        res = await respond_knowledge_node(state)

        assert res["final_answer"] == "Respuesta sin tickets de referencia."
        invoked_messages = mock_llm.ainvoke.call_args[0][0]
        # No debe haber añadido SystemMessage con referencias
        sys_ref_msg = next(
            (m for m in invoked_messages if isinstance(m, SystemMessage) and "🔗" in m.content),
            None,
        )
        assert sys_ref_msg is None


@pytest.mark.asyncio
async def test_respond_knowledge_error_sin_rag_context():
    """Caso límite: Si rag_context está vacío, debe lanzar ValueError informando falta de contexto."""
    state = {
        "messages": [HumanMessage(content="¿Cómo funciona el sistema?")],
        "rag_context": "",
    }

    with pytest.raises(ValueError, match="No se recuperó contexto del RAG"):
        await respond_knowledge_node(state)


@pytest.mark.asyncio
async def test_respond_knowledge_extrae_pregunta_desde_human_message():
    """Valida que extraiga la pregunta del último HumanMessage si user_input no está definido."""
    mock_llm_response = mock.MagicMock()
    mock_llm_response.content = "Respuesta con pregunta extraída del historial."

    mock_llm = mock.AsyncMock()
    mock_llm.ainvoke.return_value = mock_llm_response

    with mock.patch("src.agent.graph.get_llm", return_value=mock_llm):
        state = {
            "messages": [
                HumanMessage(content="Pregunta previa 1"),
                AIMessage(content="Respuesta 1"),
                HumanMessage(content="¿Pregunta relevante final?"),
            ],
            "rag_context": "Contexto para la pregunta relevante.",
        }

        res = await respond_knowledge_node(state)

        assert res["final_answer"] == "Respuesta con pregunta extraída del historial."
        mock_llm.ainvoke.assert_awaited_once()
