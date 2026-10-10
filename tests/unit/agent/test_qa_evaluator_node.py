"""
tests/unit/agent/test_qa_evaluator_node.py
=========================================
Pruebas unitarias deterministas y aisladas para qa_evaluator_node.
Evalúa la extracción estructurada de incidentes y cálculo de suficiencia hacia BugReportExtraction.
"""

from unittest import mock
import pytest

with mock.patch("src.rag.engine.RAGEngine", autospec=True):
    from src.agent.graph import qa_evaluator_node
    from src.agent.state import BugReportExtraction, State


@pytest.mark.asyncio
async def test_qa_evaluator_reporte_suficiente():
    """Valida la extracción exitosa hacia BugReportExtraction cuando el reporte cuenta con datos suficientes."""
    expected_extraction = BugReportExtraction(
        is_sufficient=True,
        title_summary="Timeout 504 en endpoint /api/v1/invoices",
        incident_type="infra_outage",
        technical_details="Spike de latencia > 30s observado en grafana a las 15:00 UTC",
        environment_info="Producción / Kubernetes clúster us-east-1",
        reproduction_steps="1. POST a /api/v1/invoices con payload > 5MB\n2. Gateway retorna 504",
        suggested_tracker="Bug",
        suggested_priority="Alta",
        clarification_questions=[],
        missing_fields=[],
    )

    mock_structured_llm = mock.AsyncMock()
    mock_structured_llm.ainvoke.return_value = expected_extraction
    mock_llm = mock.MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with mock.patch("src.agent.graph.get_llm", return_value=mock_llm):
        state = {
            "user_input": "Timeout 504 en endpoint /api/v1/invoices al enviar más de 5MB en producción",
        }
        res = await qa_evaluator_node(state)

        assert "bug_analysis" in res
        analysis: BugReportExtraction = res["bug_analysis"]
        assert analysis.is_sufficient is True
        assert analysis.title_summary == "Timeout 504 en endpoint /api/v1/invoices"
        assert analysis.incident_type == "infra_outage"
        assert analysis.suggested_priority == "Alta"
        mock_llm.with_structured_output.assert_called_once_with(BugReportExtraction)
        mock_structured_llm.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_qa_evaluator_reporte_insuficiente():
    """Valida la extracción hacia BugReportExtraction cuando faltan datos críticos para reproducir."""
    insufficient_extraction = BugReportExtraction(
        is_sufficient=False,
        title_summary="Fallo en la pantalla de facturación",
        missing_fields=["pasos para reproducir", "mensaje de error exacto"],
        clarification_questions=["¿Qué mensaje de error específico aparece en pantalla?"],
        suggested_tracker="Bug",
        suggested_priority="Normal",
    )

    mock_structured_llm = mock.AsyncMock()
    mock_structured_llm.ainvoke.return_value = insufficient_extraction
    mock_llm = mock.MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with mock.patch("src.agent.graph.get_llm", return_value=mock_llm):
        state = {
            "user_input": "La pantalla de facturación no anda",
        }
        res = await qa_evaluator_node(state)

        assert "bug_analysis" in res
        analysis: BugReportExtraction = res["bug_analysis"]
        assert analysis.is_sufficient is False
        assert "pasos para reproducir" in analysis.missing_fields
        assert len(analysis.clarification_questions) == 1
        assert "¿Qué mensaje" in analysis.clarification_questions[0]
