"""
tests/unit/agent/test_respond_existing_solution_node.py
=======================================================
Pruebas unitarias deterministas y aisladas para respond_existing_solution_node.
Evalúa la respuesta informativa al usuario cuando se detecta un incidente duplicado o previamente resuelto.
"""

from unittest import mock
import pytest
from langchain_core.messages import AIMessage

with mock.patch("src.rag.engine.RAGEngine", autospec=True):
    from src.agent.graph import respond_existing_solution_node
    from src.agent.state import State


@pytest.mark.asyncio
async def test_respond_existing_solution_con_ticket_id():
    """Valida la formulación del mensaje indicando el ticket duplicado identificado."""
    state = {
        "duplicate_issue_id": 45,
    }
    res = await respond_existing_solution_node(state)

    assert "ticket #45" in res["final_answer"]
    assert len(res["messages"]) == 1
    assert isinstance(res["messages"][0], AIMessage)
    assert res["messages"][0].content == res["final_answer"]


@pytest.mark.asyncio
async def test_respond_existing_solution_sin_ticket_id():
    """Caso de borde: duplicate_issue_id es None o ausente en el estado."""
    state = {}
    res = await respond_existing_solution_node(state)

    assert "ticket #None" in res["final_answer"]
    assert len(res["messages"]) == 1
    assert isinstance(res["messages"][0], AIMessage)
