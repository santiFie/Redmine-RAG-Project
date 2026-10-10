"""
tests/unit/agent/test_output_guardrail_node.py
==============================================
Pruebas unitarias deterministas y aisladas para output_guardrail_node.
Evalúa la validación de seguridad de la respuesta de salida con Guardrails AI (éxito, fallo, resiliencia ante excepciones).
"""

from unittest import mock
import pytest

with mock.patch("src.rag.engine.RAGEngine", autospec=True):
    from src.agent.graph import output_guardrail_node
    from src.agent.state import State


@pytest.mark.asyncio
async def test_output_guardrail_validacion_exitosa():
    """Valida que una respuesta que supera el guardrail preserve su texto validado."""
    mock_outcome = mock.MagicMock()
    mock_outcome.validation_passed = True
    mock_outcome.validated_output = "Respuesta validada y segura."

    mock_guard = mock.MagicMock()
    mock_guard.parse.return_value = mock_outcome

    with mock.patch("src.agent.graph.Guard", return_value=mock_guard):
        state = {
            "final_answer": "Respuesta original generada.",
        }

        res = await output_guardrail_node(state)

        assert res["final_answer"] == "Respuesta validada y segura."
        mock_guard.parse.assert_called_once_with(llm_output="Respuesta original generada.")


@pytest.mark.asyncio
async def test_output_guardrail_validacion_fallida():
    """Valida que una respuesta que no supera el guardrail sea reemplazada por el mensaje de seguridad predeterminado."""
    mock_outcome = mock.MagicMock()
    mock_outcome.validation_passed = False
    mock_outcome.validated_output = None

    mock_guard = mock.MagicMock()
    mock_guard.parse.return_value = mock_outcome

    with mock.patch("src.agent.graph.Guard", return_value=mock_guard):
        state = {
            "final_answer": "Texto que viola políticas de seguridad",
        }

        res = await output_guardrail_node(state)

        assert res["final_answer"] == (
            "Lo siento, la respuesta generada no superó los controles de seguridad y calidad."
        )


@pytest.mark.asyncio
async def test_output_guardrail_contenido_vacio():
    """Caso de borde: Si final_answer está vacío, retorna el estado intacto sin invocar Guard."""
    with mock.patch("src.agent.graph.Guard") as mock_guard_class:
        state = {"final_answer": ""}
        res = await output_guardrail_node(state)

        assert res == state
        mock_guard_class.assert_not_called()


@pytest.mark.asyncio
async def test_output_guardrail_tolerancia_a_excepciones():
    """Caso límite: Si Guard.parse arroja una excepción, debe preservar el contenido original sin romper el flujo."""
    mock_guard = mock.MagicMock()
    mock_guard.parse.side_effect = RuntimeError("Error en servicio de guardrails")

    with mock.patch("src.agent.graph.Guard", return_value=mock_guard):
        state = {
            "final_answer": "Contenido original a preservar ante fallo",
        }

        res = await output_guardrail_node(state)

        assert res["final_answer"] == "Contenido original a preservar ante fallo"
