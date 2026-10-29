"""
src/evaluation/e2e_evaluators.py
================================
Evaluadores LangSmith para la suite End-to-End del grafo completo.

Se centran en flujo, estado final y efectos secundarios (no repiten las métricas
por nodo de `decision_models_evaluators.py`).

Contrato de `run.outputs` (generado por tests/agent_eval/run_e2e_langsmith_evals.py):
    trajectory, final_answer, intent, is_safe_query, is_duplicate, duplicate_issue_id,
    retrieved_issue_ids, rag_context_found, target_project_id, creation_cancelled,
    create_issue_calls, created_issue_id, verified_in_redmine, interrupts, bug_analysis
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_typesafe import Noul, NoulCriteria, Question
from langsmith.evaluation.evaluator import EvaluationResult
from langsmith.schemas import Example, Run

from src.evaluation.decision_models_evaluators import get_decision_evaluator_classifier


def _outs(run: Run) -> dict[str, Any]:
    return (run.outputs or {}) if hasattr(run, "outputs") else {}


def _exp(example: Example | None) -> dict[str, Any]:
    return (example.outputs or {}) if example else {}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower())


# ==============================================================================
# 1. Trayectoria
# ==============================================================================


def e2e_trajectory_evaluator(run: Run, example: Example | None = None) -> EvaluationResult:
    """Secuencia exacta de nodos visitados vs `expected_trajectory`."""
    actual = _outs(run).get("trajectory", [])
    expected = _exp(example).get("expected_trajectory", [])
    ok = actual == expected
    return EvaluationResult(
        key="e2e_trajectory_exact_match",
        score=1.0 if ok else 0.0,
        comment=f"Esperado: {expected} | Actual: {actual}",
    )


# ==============================================================================
# 2. Efectos secundarios (creación / no creación del ticket)
# ==============================================================================


def e2e_side_effects_evaluator(run: Run, example: Example | None = None) -> EvaluationResult:
    """
    - expected_issue_created=True : exactamente 1 llamada a create_issue, proyecto correcto,
      payload con las palabras clave esperadas y (modo real) ticket verificado en Redmine.
    - expected_issue_created=False: 0 llamadas a create_issue y sin created_issue_id.
    """
    outs, exp = _outs(run), _exp(example)
    calls: list[dict[str, Any]] = outs.get("create_issue_calls", []) or []
    problems: list[str] = []

    if exp.get("expected_issue_created"):
        if len(calls) != 1:
            problems.append(f"create_issue invocado {len(calls)} veces (esperado 1)")
        else:
            call = calls[0]
            exp_proj = exp.get("expected_target_project")
            if exp_proj and str(call.get("project_id")) != exp_proj:
                problems.append(f"project_id={call.get('project_id')} (esperado {exp_proj})")
            if not (call.get("subject") or "").strip():
                problems.append("subject vacío")
            payload = _norm(f"{call.get('subject', '')} {call.get('description', '')}")
            for kw in exp.get("expected_payload_keywords", []):
                if _norm(kw) not in payload:
                    problems.append(f"falta '{kw}' en subject/description")
        if not outs.get("created_issue_id"):
            problems.append("created_issue_id ausente en el estado final")
        if exp.get("redmine_mode") == "real" and not outs.get("verified_in_redmine"):
            problems.append("el ticket no pudo verificarse en Redmine real")
    else:
        if calls:
            problems.append(f"create_issue invocado {len(calls)} veces (esperado 0)")
        if outs.get("created_issue_id"):
            problems.append(f"created_issue_id={outs.get('created_issue_id')} (esperado None)")

    return EvaluationResult(
        key="e2e_side_effects",
        score=0.0 if problems else 1.0,
        comment="OK" if not problems else "; ".join(problems),
    )


# ==============================================================================
# 3. Alineación de clarificación (HITL 1)
# ==============================================================================


def e2e_clarification_alignment_evaluator(
    run: Run, example: Example | None = None
) -> EvaluationResult:
    """
    Verifica el ciclo de clarificación:
      - se pidió (o no) clarificación según lo esperado,
      - los `missing_fields` solicitados cubren los esperados (sin faltantes ni sobrantes),
      - la respuesta del usuario terminó integrada en el análisis final (is_sufficient / keywords),
      - no se superó el máximo de turnos (anti-bucle).
    """
    outs, exp = _outs(run), _exp(example)
    expect_clar = exp.get("expected_clarification_requested", False)
    interrupts = [
        i for i in outs.get("interrupts", []) if i.get("action") == "provide_clarification"
    ]
    problems: list[str] = []

    if bool(interrupts) != expect_clar:
        problems.append(f"clarificación pedida={bool(interrupts)} (esperado {expect_clar})")
    elif expect_clar:
        if len(interrupts) > exp.get("max_clarification_turns", 1):
            problems.append(f"{len(interrupts)} turnos de clarificación (posible bucle)")
        asked = set(map(_norm, interrupts[0].get("missing_fields", [])))
        expected_fields = exp.get("expected_missing_fields", [])
        for field in expected_fields:
            if not any(_norm(field) in a or a in _norm(field) for a in asked):
                problems.append(f"no se solicitó el campo esperado '{field}'")
        if exp.get("max_extra_missing_fields") is not None:
            extra = len(asked) - len(expected_fields)
            if extra > exp["max_extra_missing_fields"]:
                problems.append(f"{extra} campos solicitados de más: {sorted(asked)}")
        if "expected_final_is_sufficient" in exp:
            final_suff = (outs.get("bug_analysis") or {}).get("is_sufficient")
            if bool(final_suff) != exp["expected_final_is_sufficient"]:
                problems.append(f"is_sufficient final={final_suff}")

    return EvaluationResult(
        key="e2e_clarification_alignment",
        score=0.0 if problems else 1.0,
        comment="OK" if not problems else "; ".join(problems),
    )


# ==============================================================================
# 4. Contrato determinístico de la respuesta final
# ==============================================================================


def e2e_final_contract_evaluator(run: Run, example: Example | None = None) -> EvaluationResult:
    """Contrato mínimo y verificable de `final_answer` según la ruta tomada."""
    outs, exp = _outs(run), _exp(example)
    answer = outs.get("final_answer", "") or ""
    problems: list[str] = []

    if not answer.strip():
        problems.append("final_answer vacío")
    for snippet in exp.get("final_answer_must_contain", []):
        if _norm(snippet) not in _norm(answer):
            problems.append(f"falta '{snippet}'")
    for snippet in exp.get("final_answer_must_not_contain", []):
        if _norm(snippet) in _norm(answer):
            problems.append(f"no debería contener '{snippet}'")
    if exp.get("expect_reference_links"):
        for iid in outs.get("retrieved_issue_ids", []):
            if f"/issues/{iid}" not in answer:
                problems.append(f"falta link al ticket #{iid}")
    if exp.get("expect_duplicate_id_in_answer") and exp.get("expected_duplicate_issue_id"):
        if str(exp["expected_duplicate_issue_id"]) not in answer:
            problems.append("no menciona el ID del ticket duplicado")
    if exp.get("expect_created_url_in_answer") and outs.get("created_issue_id"):
        if f"/issues/{outs['created_issue_id']}" not in answer:
            problems.append("no incluye la URL del ticket creado")

    return EvaluationResult(
        key="e2e_final_contract",
        score=0.0 if problems else 1.0,
        comment="OK" if not problems else "; ".join(problems),
    )


# ==============================================================================
# 5. Decision Model: ¿la respuesta omite información relevante del flujo?
# ==============================================================================

E2E_DISCLOSURE_QUESTIONS: dict[str, Question] = {
    "e2e_disclosure_completeness": Noul(
        instructions=(
            "Evalúa si la respuesta final entregada al usuario INFORMA todo lo pertinente que "
            "ocurrió en el flujo (descrito en `<hechos_del_flujo>`), sin omitir resultados que "
            "el usuario necesita conocer. No evalúes estilo ni tono, solo omisiones y fidelidad.\n\n"
            "Hechos que DEBEN comunicarse cuando aplican:\n"
            "- Consulta de conocimiento: si se encontró o NO información relevante en la base "
            "documental (y, si se encontró, qué tickets la respaldan).\n"
            "- Incidente duplicado: que el problema ya existe y en qué ticket.\n"
            "- Ticket creado: que se creó correctamente y dónde consultarlo.\n"
            "- Creación cancelada: que NO se creó ningún ticket porque el usuario lo rechazó.\n"
            "- Consulta bloqueada por seguridad: que no se puede procesar la solicitud.\n"
            "- Información no provista por el usuario tras la clarificación: que el ticket se "
            "registra igualmente con los datos disponibles, si corresponde.\n\n"
            "Penalizá también afirmaciones que CONTRADIGAN los hechos (ej. decir que se creó un "
            "ticket cuando fue cancelado, o que hay información cuando el RAG no recuperó nada).\n\n"
            "Calibración (0.0 a 1.0):\n"
            "- 0.0: Omite o contradice el resultado principal del flujo.\n"
            "- 0.5: Comunica el resultado principal pero omite un detalle pertinente "
            "(ej. no dice dónde está el ticket o no aclara que no hubo información).\n"
            "- 1.0: Comunica fielmente todos los hechos pertinentes sin contradicciones."
        ),
        criteria=NoulCriteria(
            true="La respuesta refleja fielmente todos los hechos pertinentes del flujo (puntuación 1.0).",
            false="La respuesta omite o contradice el resultado del flujo (puntuación cercana a 0.0).",
        ),
    ),
}


def _flow_facts(outs: dict[str, Any]) -> dict[str, Any]:
    calls = outs.get("create_issue_calls", []) or []
    return {
        "nodos_visitados": outs.get("trajectory", []),
        "consulta_bloqueada_por_seguridad": outs.get("is_safe_query") is False,
        "intencion": outs.get("intent"),
        "rag_recupero_contexto": outs.get("rag_context_found"),
        "tickets_recuperados_rag": outs.get("retrieved_issue_ids", []),
        "es_duplicado": outs.get("is_duplicate"),
        "ticket_duplicado_id": outs.get("duplicate_issue_id"),
        "usuario_cancelo_creacion": outs.get("creation_cancelled"),
        "ticket_creado": bool(calls) and bool(outs.get("created_issue_id")),
        "ticket_creado_id": outs.get("created_issue_id"),
        "proyecto_destino": outs.get("target_project_id"),
        "se_pidio_clarificacion": any(
            i.get("action") == "provide_clarification" for i in outs.get("interrupts", [])
        ),
    }


async def e2e_disclosure_completeness_evaluator(
    run: Run, example: Example | None = None
) -> EvaluationResult:
    """Decision Model: la respuesta final no omite ni contradice lo ocurrido en el flujo."""
    outs = _outs(run)
    inputs = run.inputs or (example.inputs if example else {})
    answer = outs.get("final_answer", "") or ""
    if not answer.strip():
        return EvaluationResult(
            key="e2e_disclosure_completeness", score=0.0, comment="final_answer vacío"
        )

    eval_state = (
        f"Mensaje original del usuario:\n{inputs.get('user_input', '')}\n\n"
        f"<hechos_del_flujo>\n{json.dumps(_flow_facts(outs), ensure_ascii=False, indent=2)}\n"
        f"</hechos_del_flujo>\n\n"
        f'Respuesta final entregada al usuario:\n"""{answer}"""'
    )
    clf = get_decision_evaluator_classifier()
    response = await clf.ainvoke({"state": eval_state, "questions": E2E_DISCLOSURE_QUESTIONS})
    ans = response.nouls.get("e2e_disclosure_completeness")
    score = round(float(ans.noul), 4) if ans else 0.0
    return EvaluationResult(
        key="e2e_disclosure_completeness",
        score=score,
        comment=f"Score JEV Completitud Informativa: {score}",
    )
