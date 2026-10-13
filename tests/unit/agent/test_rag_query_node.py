"""
tests/unit/agent/test_rag_query_node.py
=======================================
Pruebas unitarias deterministas y aisladas para rag_query_node.
Evalúa la consulta asíncrona al motor de recuperación RAG (LlamaIndex) mediante mocks.
"""

from unittest import mock
import pytest

with mock.patch("src.rag.engine.RAGEngine", autospec=True):
    from src.agent.graph import rag_query_node
    from src.agent.state import State


@pytest.mark.asyncio
async def test_rag_query_retrieval_exitoso():
    """Valida la recuperación semántica exitosa con contexto e IDs de tickets asociados."""
    mock_engine = mock.AsyncMock()
    mock_engine.aquery.return_value = {
        "context": "Contexto recuperado sobre configuración de LDAP en Redmine.",
        "issue_ids": [45, 89],
    }

    with mock.patch("src.agent.graph.get_rag_engine", return_value=mock_engine):
        state = {"user_input": "¿Cómo configurar autenticación LDAP?"}
        res = await rag_query_node(state)

        assert res["rag_context"] == "Contexto recuperado sobre configuración de LDAP en Redmine."
        assert res["retrieved_issue_ids"] == [45, 89]
        mock_engine.aquery.assert_awaited_once_with("¿Cómo configurar autenticación LDAP?")


@pytest.mark.asyncio
async def test_rag_query_retrieval_sin_resultados():
    """Caso límite: El motor RAG no encuentra documentos relevantes y retorna contexto vacío."""
    mock_engine = mock.AsyncMock()
    mock_engine.aquery.return_value = {
        "context": "",
        "issue_ids": [],
    }

    with mock.patch("src.agent.graph.get_rag_engine", return_value=mock_engine):
        state = {"user_input": "Término inexistente en la base documental"}
        res = await rag_query_node(state)

        assert res["rag_context"] == ""
        assert res["retrieved_issue_ids"] == []
        mock_engine.aquery.assert_awaited_once_with("Término inexistente en la base documental")
