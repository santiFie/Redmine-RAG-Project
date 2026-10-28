"""
src/evaluation/llm_judge_evaluators.py
======================================
Evaluador profundo generativo (LLM-as-a-Judge) para auditoría de calidad y trazabilidad
en los nodos `ask_clarification_node`, `qa_evaluator_node` y `duplicate_and_rag_check_node`.

Diseñado con instrucciones ESTRICTAMENTE ALINEADAS con las preguntas y criterios de los
Modelos de Decisión (TypeSafe JEV / Noul) de `src/evaluation/decision_models_evaluators.py`.

El prompt es AGNOSTICO a cualquier ponderación final o asimetría: el LLM únicamente
evalúa de forma objetiva las dimensiones independientes.
Para clarificación, se evalúan 5 dimensiones independientes con igual peso (0.0 a 1.0 cada una).

Soporta versionado y sincronización con el Prompt Hub de LangSmith con fallback
a plantillas locales (`src/prompts/clarification_judge_rubric.yaml`).
"""

from __future__ import annotations

import json
import os
from typing import Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client
from langsmith.evaluation.evaluator import EvaluationResult
from langsmith.schemas import Example, Run
from src.prompts.loader import get_prompt
from src.utils.get_llm import get_llm

load_dotenv()

# ==============================================================================
# 1. Configuración del Modelo Juez y Prompt Hub
# ==============================================================================

LLM_JUDGE_PROVIDER: str = os.getenv("LLM_JUDGE_PROVIDER", "openrouter")
LLM_JUDGE_MODEL: str = os.getenv("LLM_JUDGE_MODEL", "deepseek/deepseek-chat")
LLM_JUDGE_TEMP: float = float(os.getenv("LLM_JUDGE_TEMP", "0.0"))
LLM_JUDGE_PROMPT_HANDLE: str = os.getenv("LLM_JUDGE_PROMPT_HANDLE", "clarification-judge-rubric")


# ==============================================================================
# 2. Esquemas Estructurados de Auditoría (Pydantic)
# ==============================================================================


class ClarificationJudgeAudit(BaseModel):
    """
    Auditoría analítica generada por el LLM-as-a-Judge con Chain-of-Thought
    para el nodo ask_clarification_node, evaluando 5 dimensiones independientes (0.0 a 1.0 cada una).
    """

    is_empty_or_no_question: bool = Field(
        description="True si la salida del agente está vacía, no formula ninguna pregunta o no intenta clarificar el problema."
    )
    coverage_analysis: str = Field(
        description="Razonamiento paso a paso: evaluación de la cobertura de los conceptos y temas listados en missing_fields."
    )
    coverage_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Puntaje de 0.0 a 1.0 para cobertura (0.0=sin cobertura/vacía, 0.50=parcial, 1.0=cubre 100% de missing_fields).",
    )
    no_out_of_scope_analysis: str = Field(
        description="Razonamiento paso a paso: verificación de que no pregunte temas ajenos ni solicite claves, passwords o temas en must_not_ask."
    )
    no_out_of_scope_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Puntaje de 0.0 a 1.0 para restricción de alcance (0.0=pregunta credenciales/fuera de alcance, 0.50=desviación menor, 1.0=100% acotado a missing_fields).",
    )
    no_redundancy_analysis: str = Field(
        description="Razonamiento paso a paso: verificación de que no vuelva a pedir ningún dato ya informado por el usuario en user_input."
    )
    no_redundancy_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Puntaje de 0.0 a 1.0 para ausencia de redundancia (0.0=repregunta dato ya dado, 1.0=cero redundancia).",
    )
    rule_compliance_analysis: str = Field(
        description="Razonamiento paso a paso: cláusula 'no sé', entre 1 y 4 preguntas, formato numerado si >= 2, no exigir pasos en infra, etc."
    )
    rule_compliance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Puntaje de 0.0 a 1.0 para cumplimiento de reglas operativas (0.0=viola reglas críticas o falta cláusula 'no sé', 0.50=fallo menor de formato, 1.0=cumplimiento óptimo).",
    )
    tone_clarity_analysis: str = Field(
        description="Razonamiento paso a paso: empatía, lenguaje accesible no técnico, profesionalismo y concisión."
    )
    tone_clarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Puntaje de 0.0 a 1.0 para tono y claridad (0.0=hostil/vacío, 0.50=frío/verboso, 1.0=cálido, empático y conciso).",
    )
    divergence_from_decision_model_rationale: str = Field(
        default="",
        description="Explicación cualitativa detallada de por qué el caso obtuvo un score bajo (< 0.60) "
        "en la evaluación previa de Decision Models y qué falló en la generación.",
    )
    actionable_feedback: str = Field(
        description="Recomendaciones concretas y directas para ajustar el prompt de clarificación o implementar guardrails."
    )

    @property
    def min_dimension_score(self) -> float:
        """Puntaje mínimo entre las 5 dimensiones independientes."""
        return min(
            self.coverage_score,
            self.no_out_of_scope_score,
            self.no_redundancy_score,
            self.rule_compliance_score,
            self.tone_clarity_score,
        )

    @property
    def average_score(self) -> float:
        """Promedio simple de las 5 dimensiones (20% cada una)."""
        return round(
            (
                self.coverage_score
                + self.no_out_of_scope_score
                + self.no_redundancy_score
                + self.rule_compliance_score
                + self.tone_clarity_score
            )
            / 5.0,
            4,
        )

    @property
    def overall_weighted_score(self) -> float:
        """Alias para compatibilidad hacia atrás: retorna average_score."""
        return self.average_score

    @property
    def technical_focus_score(self) -> float:
        """Alias para compatibilidad hacia atrás: retorna coverage_score."""
        return self.coverage_score

    @property
    def technical_focus_analysis(self) -> str:
        """Alias para compatibilidad hacia atrás: retorna coverage_analysis."""
        return self.coverage_analysis

    @property
    def tone_conciseness_score(self) -> float:
        """Alias para compatibilidad hacia atrás: retorna tone_clarity_score."""
        return self.tone_clarity_score

    @property
    def tone_conciseness_analysis(self) -> str:
        """Alias para compatibilidad hacia atrás: retorna tone_clarity_analysis."""
        return self.tone_clarity_analysis

    def to_audit_dict(self) -> dict[str, Any]:
        """Serializa el resultado incluyendo métricas calculadas."""
        data = self.model_dump()
        data["average_score"] = self.average_score
        data["min_dimension_score"] = self.min_dimension_score
        data["overall_weighted_score"] = self.overall_weighted_score
        return data


class TriageJudgeAudit(BaseModel):
    """
    Auditoría analítica generada por el LLM-as-a-Judge para el nodo duplicate_and_rag_check_node (Capa 3).
    """

    is_reasoning_empty: bool = Field(
        description="True si el campo 'reasoning' está ausente o vacío."
    )
    rag_fidelity_analysis: str = Field(
        description="Evaluación de fidelidad al contexto RAG: verifica ausencia de alucinación de IDs de tickets o soluciones inexistentes."
    )
    rag_fidelity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Puntaje de 0.0 a 1.0 para fidelidad al contexto (1.0 = cero alucinación).",
    )
    deduplication_rules_analysis: str = Field(
        description="Evaluación de la aplicación de las 4 reglas de deduplicación (abierto, cerrado con solución, cerrado rechazado, regresión en nueva versión)."
    )
    deduplication_rules_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Puntaje de 0.0 a 1.0 para cumplimiento de reglas de deduplicación.",
    )
    project_routing_analysis: str = Field(
        description="Evaluación de la pertinencia del suggested_project_id seleccionado en función de la capa técnica afectada."
    )
    project_routing_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Puntaje de 0.0 a 1.0 para enrutamiento a proyectos.",
    )
    divergence_rationale: str = Field(
        default="",
        description="Análisis de por qué falló el caso en la evaluación de Decision Models o discrepó del Ground Truth.",
    )
    actionable_feedback: str = Field(
        description="Recomendaciones concretas para mejorar el prompt o las reglas de triaje."
    )

    @property
    def average_score(self) -> float:
        return round(
            (self.rag_fidelity_score + self.deduplication_rules_score + self.project_routing_score)
            / 3.0,
            4,
        )

    def to_audit_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["average_score"] = self.average_score
        return data


class QAExtractionJudgeAudit(BaseModel):
    """
    Auditoría analítica generada por el LLM-as-a-Judge para el nodo qa_evaluator_node (Capa 3).
    """

    completeness_analysis: str = Field(
        description="Evaluación de la completitud técnica de la extracción estructurada (BugReportExtraction)."
    )
    completeness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Puntaje de 0.0 a 1.0 para completitud de extracción.",
    )
    hallucination_analysis: str = Field(
        description="Evaluación de apego estricto a los hechos: verifica ausencia de endpoints, códigos o variables inventadas."
    )
    no_hallucination_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Puntaje de 0.0 a 1.0 para ausencia de alucinación (1.0 = cero alucinación).",
    )
    sufficiency_analysis: str = Field(
        description="Evaluación de la decisión is_sufficient según la matriz estricta de suficiencia por tipo de incidente."
    )
    sufficiency_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Puntaje de 0.0 a 1.0 para suficiencia.",
    )
    incident_type_analysis: str = Field(
        description="Evaluación de la categorización de incident_type (infra_outage, backend_api, frontend_ui, data_config, feature_request, support_request, general_bug)."
    )
    incident_type_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Puntaje de 0.0 a 1.0 para tipología de incidente.",
    )
    divergence_rationale: str = Field(
        default="",
        description="Explicación de la divergencia detectada en la evaluación previa de Decision Models.",
    )
    actionable_feedback: str = Field(
        description="Recomendaciones accionables para el prompt de qa_evaluator o esquema de extracción."
    )

    @property
    def average_score(self) -> float:
        return round(
            (
                self.completeness_score
                + self.no_hallucination_score
                + self.sufficiency_score
                + self.incident_type_score
            )
            / 4.0,
            4,
        )

    def to_audit_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["average_score"] = self.average_score
        return data


# ==============================================================================
# 3. Prompt Local de Respaldo y Carga desde LangSmith Prompt Hub
# ==============================================================================

SYSTEM_PROMPT_JUDGE = """Eres un Auditor Senior de Calidad y Juez de Evaluación Técnica (LLM-as-a-Judge) para un agente de soporte e incidencias de software.
Tu misión es realizar una auditoría cualitativa y cuantitativa imparcial, profunda y rigurosa sobre la pregunta de clarificación generada por el nodo `ask_clarification_node`.

Evalúa la respuesta del agente en cinco dimensiones independientes y objetivas (todas con el mismo peso e importancia), asignando un puntaje continuo entre 0.0 y 1.0 para cada una:

1. DIMENSIÓN 1: COBERTURA DE CAMPOS FALTANTES (coverage_score: 0.0 a 1.0)
Instrucciones: Evalúa si la pregunta formulada cubre todos los conceptos listados en missing_fields (o el ítem de fallback si missing_fields está vacío).
- 0.0: Salida vacía o no cubre ninguno de los campos faltantes.
- 0.50: Cubre parcialmente algunos campos pero omite al menos uno crítico.
- 1.0: Cubre el 100% de los temas requeridos en missing_fields.

2. DIMENSIÓN 2: RESTRICCIÓN DE ALCANCE (no_out_of_scope_score: 0.0 a 1.0)
Instrucciones: Evalúa que el agente SOLO pregunte por los temas de missing_fields. No debe inventar preguntas sobre otros temas ajenos ni solicitar contraseñas, claves, tokens o temas prohibidos (must_not_ask).
- 0.0: Indaga temas fuera de alcance o prohibidos (credenciales, datos privados, etc.).
- 0.50: Agrega alguna pregunta menor o secundaria no solicitada en missing_fields.
- 1.0: 100% acotado exclusivamente a los conceptos de missing_fields.

3. DIMENSIÓN 3: AUSENCIA DE REDUNDANCIA (no_redundancy_score: 0.0 a 1.0)
Instrucciones: Evalúa que NUNCA se vuelva a pedir un dato que el usuario ya mencionó en su reporte original o en must_not_ask.
- 0.0: Vuelve a pedir un dato explícitamente provisto en el mensaje original (ej. código HTTP 500, entorno o navegador ya dados).
- 1.0: Cero redundancia; no repregunta ningún dato ya conocido.

4. DIMENSIÓN 4: CUMPLIMIENTO DE REGLAS DE DOMINIO (rule_compliance_score: 0.0 a 1.0)
Instrucciones: Evalúa las siguientes reglas operativas:
- Cláusula anti-frustración ("No lo sé"): Debe indicar amablemente que si desconoce algún dato puede avisarlo y el ticket se registrará igual.
- Cantidad de preguntas: Entre 1 y 4 preguntas como máximo (3-4 solo si el reporte es muy vago o complejo).
- Formato: Si son >= 2 preguntas, debe usar lista numerada (1. ... 2. ...). Si es 1 sola pregunta, puede ir numerada o en texto fluido sin numerar (ninguna de las dos variantes debe penalizarse).
- Justificación técnica opcional: NO penalizar jamás la ausencia de explicaciones sobre el objetivo técnico de cada pregunta.
- En caídas de infraestructura/producción: NO exigir pasos manuales ni navegador.
- Idioma: Debe responder en el mismo idioma del reporte original.
- Calibración: 1.0 si cumple todas las reglas aplicables; 0.50 ante fallos menores de formato; 0.0 si omite la cláusula "no sé" y viola reglas críticas de formato o alcance.

5. DIMENSIÓN 5: TONO Y CLARIDAD (tone_clarity_score: 0.0 a 1.0)
Instrucciones: Evalúa empatía, tono profesional, cercano, claro para un usuario no técnico y sin verbosidad excesiva.
- 0.0: Salida vacía, hostil, condescendiente o incomprensible.
- 0.50: Tono frío, desestructurado o excesivamente verboso.
- 1.0: Tono cálido, empático, profesional, claro y con la extensión justa.
"""

USER_PROMPT_TEMPLATE = """AUDITA LA SIGUIENTE PREGUNTA DE CLARIFICACIÓN:

1. Reporte original del usuario:
{inputs}

2. Campos faltantes esperados (missing_fields):
{missing_fields}

3. Temas prohibidos o ya provistos (must_not_ask):
{must_not_ask}

4. Score previo asignado por Decision Model (JEV): {decision_model_score}

5. Pregunta generada por el agente a auditar:
\"\"\"{question_text}\"\"\"
"""


def load_judge_prompt(client: Client | None = None) -> ChatPromptTemplate:
    """
    Carga el prompt de la rúbrica del juez.
    Intenta descargarlo desde LangSmith Prompt Hub si hay cliente disponible;
    si no está disponible o falla, recurre a la plantilla YAML local en `src/prompts/clarification_judge_rubric.yaml`.
    """
    if client is not None:
        try:
            hub_prompt = client.pull_prompt(LLM_JUDGE_PROMPT_HANDLE)
            if hub_prompt is not None and isinstance(hub_prompt, ChatPromptTemplate):
                return hub_prompt
        except Exception:
            # Fallback a local ante errores de conexión o credenciales
            pass

    try:
        return get_prompt("clarification-judge-rubric")
    except Exception:
        return ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT_JUDGE),
                ("user", USER_PROMPT_TEMPLATE),
            ]
        )


def sync_judge_prompt_to_hub(
    client: Client,
    prompt_handle: str = LLM_JUDGE_PROMPT_HANDLE,
) -> str:
    """
    Publica o actualiza el prompt de la rúbrica del juez en el Prompt Hub de LangSmith.
    Retorna la URL o identificador del prompt en LangSmith.
    """
    local_prompt = get_prompt("clarification-judge-rubric")
    prompt_url = client.push_prompt(
        prompt_handle,
        object=local_prompt,
        description="Rúbrica de evaluación objetiva (LLM-as-a-Judge) para ask_clarification_node con 5 dimensiones independientes.",
        tags=["evals", "llm-judge", "clarification"],
    )
    return str(prompt_url)


# ==============================================================================
# 4. Invocadores del Evaluador LLM (Capa 3)
# ==============================================================================


def get_clarification_llm_judge(
    provider: str = LLM_JUDGE_PROVIDER,
    model: str = LLM_JUDGE_MODEL,
    temperature: float = LLM_JUDGE_TEMP,
):
    """Instancia el LLM configurado con salida estructurada según ClarificationJudgeAudit."""
    base_llm = get_llm(provider, model, temperature)
    return base_llm.with_structured_output(ClarificationJudgeAudit)


async def audit_clarification_with_llm(
    inputs: dict[str, Any],
    question_text: str,
    expected: dict[str, Any],
    decision_model_score: float | None = None,
    provider: str = LLM_JUDGE_PROVIDER,
    model: str = LLM_JUDGE_MODEL,
    client: Client | None = None,
) -> ClarificationJudgeAudit:
    """Ejecuta la auditoría del LLM-as-a-Judge para el nodo ask_clarification_node."""
    judge = get_clarification_llm_judge(provider=provider, model=model)
    prompt_template = load_judge_prompt(client=client)

    missing_fields = expected.get("missing_fields", expected.get("expected_focus_fields", []))
    must_not_ask = expected.get("must_not_ask", expected.get("prohibited_topics", []))

    format_kwargs: dict[str, Any] = {
        "inputs": json.dumps(inputs, ensure_ascii=False, indent=2),
        "missing_fields": json.dumps(missing_fields, ensure_ascii=False, indent=2),
        "must_not_ask": json.dumps(must_not_ask, ensure_ascii=False, indent=2),
        "decision_model_score": decision_model_score
        if decision_model_score is not None
        else "No disponible",
        "question_text": question_text,
    }

    # Compatibilidad si el prompt local o descargado tiene variables legadas
    if "expected_focus" in prompt_template.input_variables:
        format_kwargs["expected_focus"] = json.dumps(missing_fields, ensure_ascii=False, indent=2)
    if "prohibited_topics" in prompt_template.input_variables:
        format_kwargs["prohibited_topics"] = json.dumps(must_not_ask, ensure_ascii=False, indent=2)
    if "reference_tone" in prompt_template.input_variables:
        format_kwargs["reference_tone"] = expected.get("reference_tone", "Profesional y empático")
    if "good_example" in prompt_template.input_variables:
        format_kwargs["good_example"] = expected.get("example_good_clarification", "")

    messages = prompt_template.format_messages(**format_kwargs)
    result: ClarificationJudgeAudit = await judge.ainvoke(messages)
    return result


async def audit_triage_with_llm(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    expected: dict[str, Any],
    decision_model_score: float | None = None,
    provider: str = LLM_JUDGE_PROVIDER,
    model: str = LLM_JUDGE_MODEL,
) -> TriageJudgeAudit:
    """Ejecuta la auditoría del LLM-as-a-Judge para duplicate_and_rag_check_node (Capa 3)."""
    base_llm = get_llm(provider, model, LLM_JUDGE_TEMP)
    judge = base_llm.with_structured_output(TriageJudgeAudit)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un Auditor Senior de Triaje e Incidencias en Redmine. Tu misión es evaluar rigurosamente "
                "la decisión de deduplicación, enrutamiento a proyectos y la calidad de la justificación técnica ('reasoning') "
                "producida por el nodo de triaje frente al contexto histórico RAG y las 4 reglas de ciclo de vida.",
            ),
            (
                "user",
                "AUDITA EL SIGUIENTE CASO DE TRIAJE:\n\n"
                "1. Reporte actual (enriched_query):\n{query}\n\n"
                "2. Contexto histórico RAG:\n{rag_context}\n\n"
                "3. Proyectos disponibles:\n{projects_text}\n\n"
                "4. Decisión del agente:\n{actual_decision}\n\n"
                "5. Ground Truth esperado:\n{expected_decision}\n\n"
                "6. Score previo asignado por Decision Model (JEV): {decision_score}\n\n"
                '7. Justificación técnica (reasoning) a evaluar:\n"""{reasoning}"""',
            ),
        ]
    )

    messages = prompt.format_messages(
        query=inputs.get("enriched_query", ""),
        rag_context=inputs.get("rag_context", ""),
        projects_text=inputs.get("projects_text", ""),
        actual_decision=json.dumps(outputs, ensure_ascii=False, indent=2),
        expected_decision=json.dumps(expected, ensure_ascii=False, indent=2),
        decision_score=decision_model_score
        if decision_model_score is not None
        else "No disponible",
        reasoning=outputs.get("reasoning", ""),
    )

    result: TriageJudgeAudit = await judge.ainvoke(messages)
    return result


async def audit_qa_extraction_with_llm(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    expected: dict[str, Any],
    decision_model_score: float | None = None,
    provider: str = LLM_JUDGE_PROVIDER,
    model: str = LLM_JUDGE_MODEL,
) -> QAExtractionJudgeAudit:
    """Ejecuta la auditoría del LLM-as-a-Judge para qa_evaluator_node (Capa 3)."""
    base_llm = get_llm(provider, model, LLM_JUDGE_TEMP)
    judge = base_llm.with_structured_output(QAExtractionJudgeAudit)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un Auditor Senior de Calidad (QA) y Soporte Técnico en Redmine. Tu objetivo es auditar la extracción "
                "estructurada de incidentes generada por qa_evaluator_node, evaluando completitud, ausencia de alucinación, "
                "suficiencia según la matriz de dominio y la correcta clasificación de incident_type.",
            ),
            (
                "user",
                "AUDITA LA SIGUIENTE EXTRACCIÓN DE QA:\n\n"
                "1. Input original del usuario:\n{user_input}\n\n"
                "2. Extracción generada por el agente:\n{actual_extraction}\n\n"
                "3. Ground Truth esperado:\n{expected_extraction}\n\n"
                "4. Score previo asignado por Decision Model (JEV): {decision_score}",
            ),
        ]
    )

    messages = prompt.format_messages(
        user_input=json.dumps(inputs, ensure_ascii=False, indent=2),
        actual_extraction=json.dumps(outputs, ensure_ascii=False, indent=2),
        expected_extraction=json.dumps(expected, ensure_ascii=False, indent=2),
        decision_score=decision_model_score
        if decision_model_score is not None
        else "No disponible",
    )

    result: QAExtractionJudgeAudit = await judge.ainvoke(messages)
    return result


async def clarification_llm_judge_evaluator(
    run: Run,
    example: Example | None = None,
) -> EvaluationResult:
    """Evaluador LangSmith compatible para auditoría selectiva con LLM-as-a-Judge."""
    outputs = run.outputs or {}
    expected = example.outputs if example else {}
    inputs = run.inputs or (example.inputs if example else {})

    question_text = ""
    if isinstance(outputs, dict):
        messages = outputs.get("messages", [])
        if messages and isinstance(messages, list):
            for msg in messages:
                if hasattr(msg, "content") and (
                    getattr(msg, "type", "") == "ai" or msg.__class__.__name__ == "AIMessage"
                ):
                    question_text = str(getattr(msg, "content", ""))
                    break
            if not question_text and messages:
                question_text = str(getattr(messages[0], "content", messages[0]))
        if not question_text:
            question_text = str(outputs.get("question", outputs.get("question_text", "")) or "")
    elif isinstance(outputs, str):
        question_text = outputs

    audit = await audit_clarification_with_llm(
        inputs=inputs,
        question_text=question_text,
        expected=expected,
    )

    return EvaluationResult(
        key="llm_judge_clarification_audit",
        score=audit.average_score,
        comment=(
            f"[LLM Judge] Score Promedio: {audit.average_score} (Mínimo: {audit.min_dimension_score})\n"
            f"Cobertura ({audit.coverage_score}): {audit.coverage_analysis}\n"
            f"Alcance ({audit.no_out_of_scope_score}): {audit.no_out_of_scope_analysis}\n"
            f"Redundancia ({audit.no_redundancy_score}): {audit.no_redundancy_analysis}\n"
            f"Reglas ({audit.rule_compliance_score}): {audit.rule_compliance_analysis}\n"
            f"Tono/Claridad ({audit.tone_clarity_score}): {audit.tone_clarity_analysis}\n"
            f"Recomendación: {audit.actionable_feedback}"
        ),
        extra=audit.to_audit_dict(),
    )


__all__ = [
    "ClarificationJudgeAudit",
    "TriageJudgeAudit",
    "QAExtractionJudgeAudit",
    "LLM_JUDGE_PROMPT_HANDLE",
    "SYSTEM_PROMPT_JUDGE",
    "load_judge_prompt",
    "sync_judge_prompt_to_hub",
    "get_clarification_llm_judge",
    "audit_clarification_with_llm",
    "audit_triage_with_llm",
    "audit_qa_extraction_with_llm",
    "clarification_llm_judge_evaluator",
]
