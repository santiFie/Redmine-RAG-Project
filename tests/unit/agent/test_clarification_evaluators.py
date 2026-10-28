"""
tests/unit/agent/test_clarification_evaluators.py
=================================================
Pruebas unitarias para evaluadores de clarificación, suficiencia y triaje:
- Capa 1 determinista (salida vacía -> score 0.0 en las 5 dimensiones sin llamar a JEV).
- Esquema de 5 preguntas atómicas en CLARIFICATION_QUESTIONS.
- 5 evaluadores independientes de clarificación (100% c/u) con caché por run.id.
- Evaluador de incident_type con soporte de 'feature_request' y 'support_request'.
- Evaluador de suficiencia (sufficiency_decision_evaluator).
- Evaluador de razonamiento de triaje (triage_reasoning_evaluator).
- Validación de esquemas LLM-as-a-Judge (ClarificationJudgeAudit, TriageJudgeAudit, QAExtractionJudgeAudit).
- Carga de prompt agnóstico de 5 dimensiones.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from langsmith.schemas import Example, Run

from src.evaluation.decision_models_evaluators import (
    CLARIFICATION_QUESTIONS,
    CLARIFICATION_RELEVANCE_QUESTIONS,
    INCIDENT_TYPE_QUESTIONS,
    TRIAGE_REASONING_QUESTIONS,
    _CLARIFICATION_EVAL_CACHE,
    clarification_coverage_evaluator,
    clarification_no_out_of_scope_evaluator,
    clarification_no_redundancy_evaluator,
    clarification_relevance_evaluator,
    clarification_rule_compliance_evaluator,
    clarification_technical_evaluator,
    clarification_tone_clarity_evaluator,
    clarification_tone_evaluator,
    incident_type_evaluator,
    sufficiency_decision_evaluator,
    triage_reasoning_evaluator,
)
from src.evaluation.llm_judge_evaluators import (
    ClarificationJudgeAudit,
    QAExtractionJudgeAudit,
    TriageJudgeAudit,
    load_judge_prompt,
)


@pytest.fixture(autouse=True)
def clear_eval_cache():
    """Limpia el caché de evaluaciones antes de cada test."""
    _CLARIFICATION_EVAL_CACHE.clear()
    yield
    _CLARIFICATION_EVAL_CACHE.clear()


@pytest.mark.asyncio
async def test_capa_1_salida_vacia_retorna_score_cero_en_las_5_metricas():
    """Verifica que si la salida está vacía, retorna 0.0 determinista en las 5 métricas sin invocar a JEV."""
    run = Run(
        id="00000000-0000-0000-0000-000000000001",
        name="test_empty",
        inputs={"user_input": "Error 500 al guardar"},
        outputs={"question": "", "missing_fields": ["campo1"]},
        run_type="chain",
        start_time=datetime.now(timezone.utc),
    )
    example = Example(
        id="00000000-0000-0000-0000-000000000002",
        dataset_id="00000000-0000-0000-0000-000000000003",
        inputs={"user_input": "Error 500 al guardar"},
        outputs={"missing_fields": ["campo1"]},
        created_at=datetime.now(timezone.utc),
        modified_at=datetime.now(timezone.utc),
    )

    with patch(
        "src.evaluation.decision_models_evaluators.get_decision_evaluator_classifier"
    ) as mock_clf:
        cov_res = await clarification_coverage_evaluator(run, example)
        scope_res = await clarification_no_out_of_scope_evaluator(run, example)
        redun_res = await clarification_no_redundancy_evaluator(run, example)
        rule_res = await clarification_rule_compliance_evaluator(run, example)
        tone_res = await clarification_tone_clarity_evaluator(run, example)

        # No debe haber llamado al clasificador JEV en ninguna ocasión (Costo 0)
        mock_clf.assert_not_called()
        assert cov_res.score == 0.0
        assert scope_res.score == 0.0
        assert redun_res.score == 0.0
        assert rule_res.score == 0.0
        assert tone_res.score == 0.0


def test_clarification_questions_schema_5_dimensiones():
    """Valida que las 5 preguntas atómicas existan y contengan instrucciones calibradas."""
    expected_dims = [
        "clarification_coverage",
        "clarification_no_out_of_scope",
        "clarification_no_redundancy",
        "clarification_rule_compliance",
        "clarification_tone_clarity",
    ]
    for dim in expected_dims:
        assert dim in CLARIFICATION_QUESTIONS, (
            f"Falta la dimensión {dim} en CLARIFICATION_QUESTIONS"
        )

    assert CLARIFICATION_RELEVANCE_QUESTIONS == CLARIFICATION_QUESTIONS

    rule_q = CLARIFICATION_QUESTIONS["clarification_rule_compliance"]
    assert "Cláusula 'No sé'" in rule_q.instructions
    assert "Cantidad de preguntas" in rule_q.instructions
    assert "Formato de presentación" in rule_q.instructions


@pytest.mark.asyncio
async def test_5_evaluadores_independientes_y_cache_por_run_id():
    """
    Verifica que los 5 evaluadores retornen sus scores independientes al 100%
    y que gracias al caché por run.id se realice EXACTAMENTE 1 sola llamada a JEV.
    """
    run_id = "00000000-0000-0000-0000-000000000010"
    run = Run(
        id=run_id,
        name="test_5_dims",
        inputs={"user_input": "Falla pago al guardar", "missing_fields": ["navegador", "error"]},
        outputs={
            "question": "1. ¿En qué navegador ocurre?\n2. ¿Qué error muestra?\nSi no lo sabes indícanos."
        },
        run_type="chain",
        start_time=datetime.now(timezone.utc),
    )
    example = Example(
        id="00000000-0000-0000-0000-000000000020",
        dataset_id="00000000-0000-0000-0000-000000000030",
        inputs={"user_input": "Falla pago al guardar", "missing_fields": ["navegador", "error"]},
        outputs={"missing_fields": ["navegador", "error"], "must_not_ask": ["reiniciar router"]},
        created_at=datetime.now(timezone.utc),
        modified_at=datetime.now(timezone.utc),
    )

    mock_response = MagicMock()
    mock_response.nouls = {
        "clarification_coverage": MagicMock(noul=0.90),
        "clarification_no_out_of_scope": MagicMock(noul=1.00),
        "clarification_no_redundancy": MagicMock(noul=1.00),
        "clarification_rule_compliance": MagicMock(noul=0.85),
        "clarification_tone_clarity": MagicMock(noul=0.75),
    }

    mock_client = MagicMock()
    mock_client.ainvoke = AsyncMock(return_value=mock_response)

    with patch(
        "src.evaluation.decision_models_evaluators.get_decision_evaluator_classifier",
        return_value=mock_client,
    ):
        # Invocamos los 5 evaluadores independientes
        cov_res = await clarification_coverage_evaluator(run, example)
        scope_res = await clarification_no_out_of_scope_evaluator(run, example)
        redun_res = await clarification_no_redundancy_evaluator(run, example)
        rule_res = await clarification_rule_compliance_evaluator(run, example)
        tone_res = await clarification_tone_clarity_evaluator(run, example)

        # Cada evaluador debe reportar su propio score al 100%
        assert cov_res.score == 0.90
        assert scope_res.score == 1.00
        assert redun_res.score == 1.00
        assert rule_res.score == 0.85
        assert tone_res.score == 0.75

        # Invocamos el evaluador de compatibilidad (promedio)
        rel_res = await clarification_relevance_evaluator(run, example)
        # Promedio: (0.90 + 1.0 + 1.0 + 0.85 + 0.75) / 5 = 4.5 / 5 = 0.90
        assert rel_res.score == 0.90

        # Verificamos que ainvoke se haya llamado EXACTAMENTE UNA SOLA VEZ para las 6 invocaciones
        assert mock_client.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_incident_type_evaluator_soporta_feature_y_support_request():
    """Verifica que incident_type_evaluator soporte 'feature_request' y 'support_request'."""
    assert "feature_request" in INCIDENT_TYPE_QUESTIONS["incident_type"].criteria
    assert "support_request" in INCIDENT_TYPE_QUESTIONS["incident_type"].criteria

    # Caso Feature Request Coincidente
    example_feat = Example(
        id="00000000-0000-0000-0000-000000000101",
        dataset_id="00000000-0000-0000-0000-000000000102",
        inputs={"user_input": "Quisiera agregar botón de exportación a Excel"},
        outputs={"incident_type": "feature_request"},
        created_at=datetime.now(timezone.utc),
        modified_at=datetime.now(timezone.utc),
    )
    run_feat = Run(
        id="00000000-0000-0000-0000-000000000103",
        name="test_feat",
        inputs={"user_input": "Quisiera exportar a Excel"},
        outputs={"incident_type": "feature_request"},
        run_type="chain",
        start_time=datetime.now(timezone.utc),
    )
    res_feat = await incident_type_evaluator(run_feat, example_feat)
    assert res_feat.score == 1.0
    assert res_feat.extra["is_match"] is True

    # Caso Support Request Coincidente
    example_sup = Example(
        id="00000000-0000-0000-0000-000000000201",
        dataset_id="00000000-0000-0000-0000-000000000202",
        inputs={"user_input": "Usuario no puede acceder a reportes (403)"},
        outputs={"incident_type": "support_request"},
        created_at=datetime.now(timezone.utc),
        modified_at=datetime.now(timezone.utc),
    )
    run_sup = Run(
        id="00000000-0000-0000-0000-000000000203",
        name="test_sup",
        inputs={"user_input": "Usuario 403"},
        outputs={"incident_type": "support_request"},
        run_type="chain",
        start_time=datetime.now(timezone.utc),
    )
    res_sup = await incident_type_evaluator(run_sup, example_sup)
    assert res_sup.score == 1.0

    # Caso Discrepancia
    run_bug = Run(
        id="00000000-0000-0000-0000-000000000204",
        name="test_mismatch",
        inputs={"user_input": "Usuario 403"},
        outputs={"incident_type": "general_bug"},
        run_type="chain",
        start_time=datetime.now(timezone.utc),
    )
    res_mismatch = await incident_type_evaluator(run_bug, example_sup)
    assert res_mismatch.score == 0.0
    assert res_mismatch.extra["is_match"] is False


@pytest.mark.asyncio
async def test_sufficiency_decision_evaluator():
    """Valida el evaluador determinista de suficiencia (qa_evaluator_node)."""
    example_suff = Example(
        id="00000000-0000-0000-0000-000000000301",
        dataset_id="00000000-0000-0000-0000-000000000302",
        inputs={"user_input": "Error con datos suficientes"},
        outputs={"is_sufficient": True},
        created_at=datetime.now(timezone.utc),
        modified_at=datetime.now(timezone.utc),
    )

    # Coincidencia is_sufficient=True
    run_true = Run(
        id="00000000-0000-0000-0000-000000000303",
        name="test_suff_true",
        inputs={},
        outputs={"is_sufficient": True},
        run_type="chain",
        start_time=datetime.now(timezone.utc),
    )
    res_true = await sufficiency_decision_evaluator(run_true, example_suff)
    assert res_true.score == 1.0
    assert res_true.extra["is_match"] is True

    # Discrepancia
    run_false = Run(
        id="00000000-0000-0000-0000-000000000304",
        name="test_suff_false",
        inputs={},
        outputs={"is_sufficient": False},
        run_type="chain",
        start_time=datetime.now(timezone.utc),
    )
    res_false = await sufficiency_decision_evaluator(run_false, example_suff)
    assert res_false.score == 0.0
    assert res_false.extra["is_match"] is False


@pytest.mark.asyncio
async def test_triage_reasoning_evaluator():
    """Valida el evaluador de razonamiento de triaje (duplicate_and_rag_check_node)."""
    assert "triage_reasoning_quality" in TRIAGE_REASONING_QUESTIONS

    # Caso razonamiento vacío -> 0.0 sin llamar a JEV
    run_empty = Run(
        id="00000000-0000-0000-0000-000000000401",
        name="test_reasoning_empty",
        inputs={"enriched_query": "Falla", "rag_context": "Contexto"},
        outputs={"reasoning": "", "is_duplicate": False},
        run_type="chain",
        start_time=datetime.now(timezone.utc),
    )
    with patch(
        "src.evaluation.decision_models_evaluators.get_decision_evaluator_classifier"
    ) as mock_clf:
        res_empty = await triage_reasoning_evaluator(run_empty)
        mock_clf.assert_not_called()
        assert res_empty.score == 0.0
        assert res_empty.extra["is_empty"] is True

    # Caso razonamiento con contenido -> evalúa con JEV
    run_valid = Run(
        id="00000000-0000-0000-0000-000000000402",
        name="test_reasoning_valid",
        inputs={"enriched_query": "Falla en pagos", "rag_context": "Ticket #10: Error pago"},
        outputs={
            "reasoning": "El ticket #10 está abierto con la misma causa raíz.",
            "is_duplicate": True,
            "duplicate_issue_id": 10,
            "suggested_project_id": "pagos",
        },
        run_type="chain",
        start_time=datetime.now(timezone.utc),
    )
    mock_resp = MagicMock()
    mock_resp.nouls = {"triage_reasoning_quality": MagicMock(noul=0.95)}
    mock_clf_client = MagicMock()
    mock_clf_client.ainvoke = AsyncMock(return_value=mock_resp)

    with patch(
        "src.evaluation.decision_models_evaluators.get_decision_evaluator_classifier",
        return_value=mock_clf_client,
    ):
        res_valid = await triage_reasoning_evaluator(run_valid)
        assert res_valid.score == 0.95
        assert res_valid.extra["is_empty"] is False
        mock_clf_client.ainvoke.assert_awaited_once()


def test_llm_judge_schema_validation_5_dimensiones():
    """
    Verifica que ClarificationJudgeAudit valide las 5 dimensiones independientes,
    calcule min_dimension_score y average_score correctamente.
    """
    audit = ClarificationJudgeAudit(
        is_empty_or_no_question=False,
        coverage_analysis="Cubre todos los campos faltantes requeridos.",
        coverage_score=0.90,
        no_out_of_scope_analysis="No pregunta nada fuera de alcance ni credenciales.",
        no_out_of_scope_score=1.00,
        no_redundancy_analysis="Cero redundancia con el input original.",
        no_redundancy_score=1.00,
        rule_compliance_analysis="Incluye cláusula 'no sé' y tiene 2 preguntas numeradas.",
        rule_compliance_score=0.85,
        tone_clarity_analysis="Tono amable, empático y conciso.",
        tone_clarity_score=0.75,
        divergence_from_decision_model_rationale="",
        actionable_feedback="Ninguna corrección necesaria.",
    )

    # 5 dimensiones validadas
    assert audit.coverage_score == 0.90
    assert audit.no_out_of_scope_score == 1.00
    assert audit.no_redundancy_score == 1.00
    assert audit.rule_compliance_score == 0.85
    assert audit.tone_clarity_score == 0.75

    # Métricas calculadas
    assert audit.min_dimension_score == 0.75
    # (0.90 + 1.0 + 1.0 + 0.85 + 0.75) / 5 = 4.5 / 5 = 0.90
    assert audit.average_score == 0.90
    assert audit.overall_weighted_score == 0.90

    audit_dict = audit.to_audit_dict()
    assert audit_dict["average_score"] == 0.90
    assert audit_dict["min_dimension_score"] == 0.75


def test_triage_judge_audit_and_qa_extraction_judge_audit_schemas():
    """Valida los esquemas de auditoría de Capa 3 para Triaje y QA Extraction."""
    triage_audit = TriageJudgeAudit(
        is_reasoning_empty=False,
        rag_fidelity_analysis="Cita fielmente el ticket #42 del contexto.",
        rag_fidelity_score=1.0,
        deduplication_rules_analysis="Aplica Regla 1 para ticket abierto.",
        deduplication_rules_score=0.9,
        project_routing_analysis="Selecciona el proyecto de pagos adecuadamente.",
        project_routing_score=0.8,
        divergence_rationale="",
        actionable_feedback="Ninguno.",
    )
    # (1.0 + 0.9 + 0.8) / 3 = 0.9
    assert triage_audit.average_score == 0.9

    qa_audit = QAExtractionJudgeAudit(
        completeness_analysis="Extracción completa de endpoints y error.",
        completeness_score=0.9,
        hallucination_analysis="Cero alucinación.",
        no_hallucination_score=1.0,
        sufficiency_analysis="Suficiencia correcta según matriz.",
        sufficiency_score=1.0,
        incident_type_analysis="backend_api clasificado correctamente.",
        incident_type_score=1.0,
        divergence_rationale="",
        actionable_feedback="Ninguno.",
    )
    # (0.9 + 1.0 + 1.0 + 1.0) / 4 = 0.975
    assert qa_audit.average_score == 0.975


def test_load_judge_prompt_5_dimensiones():
    """Verifica que el prompt de la rúbrica cargue correctamente y contenga las 5 dimensiones."""
    prompt = load_judge_prompt(client=None)
    assert prompt is not None
    messages_str = str(prompt)
    assert "COBERTURA DE CAMPOS FALTANTES" in messages_str
    assert "RESTRICCIÓN DE ALCANCE" in messages_str
    assert "AUSENCIA DE REDUNDANCIA" in messages_str
    assert "CUMPLIMIENTO DE REGLAS DE DOMINIO" in messages_str
    assert "TONO Y CLARIDAD" in messages_str
