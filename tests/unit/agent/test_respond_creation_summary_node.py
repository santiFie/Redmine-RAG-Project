"""
tests/unit/agent/test_respond_creation_summary_node.py
======================================================
Pruebas unitarias deterministas y aisladas para respond_creation_summary_node.
Evalúa el mensaje de confirmación entregado al usuario con el enlace directo al ticket creado en Redmine.
"""

from unittest import mock
import pytest
from langchain_core.messages import AIMessage

with mock.patch("src.rag.engine.RAGEngine", autospec=True):
    from src.agent.graph import respond_creation_summary_node
    from src.agent.state import State


@pytest.mark.asyncio
async def test_respond_creation_summary_con_url_valida():
    """Valida la generación del mensaje con el link directo al nuevo ticket creado."""
    state = {
        "created_issue_url": "http://redmine.corp:3000/issues/999",
    }
    res = await respond_creation_summary_node(state)

    assert "http://redmine.corp:3000/issues/999" in res["final_answer"]
    assert len(res["messages"]) == 1
    assert isinstance(res["messages"][0], AIMessage)
    assert res["messages"][0].content == res["final_answer"]


@pytest.mark.asyncio
async def test_respond_creation_summary_sin_url():
    """Caso de borde: created_issue_url no está definido en el estado."""
    state = {}
    res = await respond_creation_summary_node(state)

    assert ": #" in res["final_answer"]
    assert len(res["messages"]) == 1
    assert isinstance(res["messages"][0], AIMessage)
