"""
src/evaluation/decision_models_evaluators.py
============================================
Centralización de configuraciones, modelos de decisión, esquemas de preguntas
(Noul y Choice) y funciones evaluadoras para LangSmith utilizando TypeSafe JEV
(vía OpenRouter) para evaluar los nodos LLM del agente.

Diseñado con el mismo patrón declarativo que `src/agent/typesafe_config.py`.
"""

from __future__ import annotations

import json
import os
from typing import Any
from dotenv import load_dotenv
from pydantic import SecretStr

from langchain_typesafe import Choice, Noul, NoulCriteria, Question, TypeSafeClassifier
from langsmith.evaluation.evaluator import EvaluationResult
from langsmith.schemas import Example, Run

load_dotenv()

# ==============================================================================
# 1. Configuración de Conexión a OpenRouter / TypeSafe
# ==============================================================================

OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api")
DECISION_EVAL_MODEL: str = os.getenv("DECISION_EVAL_MODEL", "typesafe/jev-1.13")


def get_decision_evaluator_classifier(
    model: str = DECISION_EVAL_MODEL,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> TypeSafeClassifier:
    """Instancia el cliente TypeSafeClassifier para los evaluadores de decisión."""
    resolved_base_url = (base_url or OPENROUTER_BASE_URL).rstrip("/")
    resolved_api_key = (
        api_key
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("TYPESAFE_API_KEY")
        or "dummy-eval-key"
    )
    return TypeSafeClassifier(
        model=model,
        base_url=resolved_base_url,
        api_key=SecretStr(resolved_api_key) if resolved_api_key else None,
        timeout=timeout,
    )


# ==============================================================================
# 2. Definiciones de Preguntas y Criterios con Few-Shot Calibrados
# ==============================================================================

# ------------------------------------------------------------------------------
# A. extraction_completeness (Noul continuo 0.0 .. 1.0)
# ------------------------------------------------------------------------------
EXTRACTION_COMPLETENESS_QUESTIONS: dict[str, Question] = {
    "extraction_completeness": Noul(
        instructions=(
            "Evalúa el nivel de completitud de la extracción estructurada realizada sobre el incidente reportado "
            "en comparación con la información provista en el input y el ground truth.\n\n"
            "Guía de calibración de puntuación (0.0 a 1.0):\n"
            "- 0.0 (Mínimo / Totalmente incompleta): La extracción omitió todos o casi todos los datos críticos. "
            "Ejemplo: El reporte detalla endpoint, código 500 y stacktrace, pero la extracción deja campos vacíos y no identificó el problema.\n"
            "- 0.25 (Pobre / Omisión de datos críticos): Se capturó un título genérico, pero se perdieron datos vitales presentes en el mensaje, "
            "como el endpoint exacto o el stacktrace.\n"
            "- 0.50 (Aceptable / Parcial): Se extrajo el núcleo del problema y el síntoma principal, pero faltaron detalles secundarios como "
            "el entorno (versión/navegador) o pasos de reproducción que sí estaban presentes en el texto.\n"
            "- 0.75 (Buena / Casi completa): Se extrajeron todos los elementos clave (título, pasos, mensaje de error y entorno), omitiendo "
            "únicamente matices menores o de formato no esenciales.\n"
            "- 1.0 (Óptima / Completa): Se capturaron exhaustivamente todos y cada uno de los datos presentes en el input "
            "sin omisiones de ningún detalle técnico, entorno o pasos indicados."
        ),
        criteria=NoulCriteria(
            true=(
                "La extracción es fiel y completa respecto al mensaje original: captura el error, endpoints, "
                "entorno, pasos de reproducción y metadatos relevantes sin pérdidas sustanciales (puntuación cercana a 1.0)."
            ),
            false=(
                "La extracción es insuficiente, truncada o pierde información técnica indispensable para dar seguimiento "
                "o reproducir el incidente (puntuación cercana a 0.0)."
            ),
        ),
    ),
}

# ------------------------------------------------------------------------------
# B. no_hallucination (Noul continuo 0.0 .. 1.0)
# ------------------------------------------------------------------------------
NO_HALLUCINATION_QUESTIONS: dict[str, Question] = {
    "no_hallucination": Noul(
        instructions=(
            "Evalúa si la información estructurada extraída se apega estrictamente al texto del usuario o si contiene "
            "alucinaciones (datos técnicos, URLs, endpoints, parámetros o pasos inventados que no se desprenden del input).\n\n"
            "Guía de calibración de puntuación (0.0 a 1.0):\n"
            "- 0.0 (Alucinación severa / Fabricación): Se inventaron endpoints, tablas de base de datos o stacktraces que no existían "
            "en el reporte. Ejemplo: El usuario reportó 'falla el checkout' y la extracción inventa '/api/v2/orders?discount=true' y 'tabla order_items corrupta'.\n"
            "- 0.25 (Alucinación alta): Se inventaron detalles técnicos significativos como códigos de estado específicos (ej. 403 Forbidden) "
            "o tecnologías no mencionadas que alteran la causa raíz.\n"
            "- 0.50 (Alucinación moderada / Asunciones no verificadas): Se agregaron suposiciones de entorno plausibles pero no afirmadas "
            "por el usuario (ej. asumir navegador Chrome en macOS cuando el usuario no especificó plataforma).\n"
            "- 0.75 (Prácticamente libre de alucinación): Todo el contenido técnico es verídico, con mínimas inferencias naturales de redacción "
            "que no introducen información técnica falsa.\n"
            "- 1.0 (Completamente veraz / Cero alucinación): Absolutamente cada afirmación, dato técnico, paso y entorno se deriva "
            "directamente y de forma comprobable del mensaje original."
        ),
        criteria=NoulCriteria(
            true=(
                "Cero alucinación: toda la información presente en la extracción proviene directamente o es una paráfrasis "
                "estricta y exacta de los hechos aportados por el usuario (puntuación cercana a 1.0)."
            ),
            false=(
                "Presencia de alucinaciones o datos fabricados que no estaban en el reporte del usuario (puntuación cercana a 0.0)."
            ),
        ),
    ),
}

# ------------------------------------------------------------------------------
# C. incident_type (Choice categórico)
# ------------------------------------------------------------------------------
INCIDENT_TYPE_QUESTIONS: dict[str, Question] = {
    "incident_type": Choice(
        instructions=(
            "Clasifica la tipología técnica del incidente reportado entre las categorías permitidas, "
            "considerando la naturaleza de la falla según los ejemplos descritos en los criterios."
        ),
        criteria={
            "infra_outage": (
                "Caídas de infraestructura, saturación de hardware, red o base de datos.\n"
                "Ejemplos: 'PostgreSQL no acepta conexiones por max_connections', 'OOMKilled 137 en pods de Kubernetes', "
                "'Timeouts 504 de Nginx/Cloudflare por nodo caído', 'Pool saturado en pgBouncer'."
            ),
            "backend_api": (
                "Errores de servidor, excepciones de código backend, respuestas HTTP 5xx/4xx de endpoints o fallos en lógica de negocio.\n"
                "Ejemplos: 'POST /api/v1/checkout responde 500 KeyError billing_address', 'Falla la validación de token JWT en el middleware', "
                "'NullPointerException al procesar webhook de Stripe'."
            ),
            "frontend_ui": (
                "Fallas visuales, comportamientos inesperados en la interfaz de usuario, CSS o JavaScript del navegador/móvil.\n"
                "Ejemplos: 'El botón de pago queda deshabilitado en Safari iOS', 'El menú desplegable queda oculto detrás de un iframe', "
                "'Error Uncaught TypeError en bundle.js al hacer click en filtros'."
            ),
            "data_config": (
                "Inconsistencias de datos, registros corruptos, errores de configuración de variables o parámetros del sistema.\n"
                "Ejemplos: 'Un cliente tiene saldo negativo por registro huérfano', 'Variable REDMINE_URL mal configurada en staging', "
                "'Falta la columna active_flag en la migración de testing'."
            ),
            "feature_request": (
                "Solicitudes de nuevas funcionalidades, mejoras de producto, nuevas opciones de exportación o automatizaciones que no son errores actuales.\n"
                "Ejemplos: 'Agregar botón para exportar a Excel', 'Permitir adjuntar múltiples PDFs en el checkout', 'Crear nuevo rol de auditor'."
            ),
            "support_request": (
                "Solicitudes operativas de soporte, problemas de permisos/accesos de usuarios (ej. HTTP 403 teniendo rol asignado), blanqueo de claves o consultas operativas.\n"
                "Ejemplos: 'El usuario juan@empresa.com no puede acceder al módulo de facturación a pesar de tener rol contable', "
                "'Solicito resetear contraseña del usuario admin', 'HTTP 403 Forbidden al ver reportes asignados'."
            ),
            "general_bug": (
                "Bugs funcionales genéricos o reportes con contexto insuficiente donde aún no se puede determinar la capa técnica afectada.\n"
                "Ejemplos: 'El sistema no guarda los cambios', 'No anda la exportación', 'Da error al intentar finalizar el proceso'."
            ),
        },
    ),
}

# ------------------------------------------------------------------------------
# D. sufficiency_decision (Noul / Determinista de Suficiencia)
# ------------------------------------------------------------------------------
SUFFICIENCY_DECISION_QUESTIONS: dict[str, Question] = {
    "sufficiency_decision": Noul(
        instructions=(
            "Evalúa si el reporte contiene información suficiente para que el equipo técnico pueda actuar "
            "o investigar el problema según la matriz de suficiencia por tipo de incidente.\n\n"
            "Guía de calibración de puntuación (0.0 a 1.0):\n"
            "- 0.0: Información totalmente insuficiente; omite datos críticos indispensables sin los cuales no es posible proceder.\n"
            "- 1.0: Información suficiente según los requisitos mínimos de su categoría técnica."
        ),
        criteria=NoulCriteria(
            true="El reporte cumple con los requisitos mínimos de suficiencia de su tipología técnica (score 1.0).",
            false="El reporte carece de datos críticos indispensables para iniciar el triaje o reproducción (score 0.0).",
        ),
    ),
}

# ------------------------------------------------------------------------------
# E. clarification_questions (5 Dimensiones Atómicas 100% Desacopladas)
# ------------------------------------------------------------------------------
CLARIFICATION_QUESTIONS: dict[str, Question] = {
    "clarification_coverage": Noul(
        instructions=(
            "Evalúa si las preguntas formuladas por el agente cubren todos los conceptos y temas "
            "listados en `<campos_faltantes>` (o el texto de fallback si la lista estaba vacía).\n\n"
            "Guía de calibración de puntuación (0.0 a 1.0):\n"
            "- 0.0 (Sin cobertura / Vacío): La salida está vacía o no formula preguntas sobre ninguno de los campos faltantes requeridos.\n"
            "- 0.50 (Cobertura parcial): Cubre algunos campos faltantes pero omite al menos uno de los temas requeridos.\n"
            "- 1.0 (Cobertura completa): Formula preguntas claras que cubren el 100% de los temas indicados en `<campos_faltantes>`."
        ),
        criteria=NoulCriteria(
            true="Las preguntas generadas indagan exhaustivamente todos los temas requeridos en campos_faltantes (puntuación 1.0).",
            false="Se omiten campos faltantes requeridos o la salida está vacía (puntuación cercana a 0.0).",
        ),
    ),
    "clarification_no_out_of_scope": Noul(
        instructions=(
            "Evalúa que el agente formule preguntas EXCLUSIVAMENTE sobre los conceptos indicados en `<campos_faltantes>`, "
            "sin desviarse hacia temas ajenos ni solicitar jamás datos confidenciales, contraseñas, claves privadas o tokens "
            "(respetando estrictamente `must_not_ask` si está definido).\n\n"
            "Guía de calibración de puntuación (0.0 a 1.0):\n"
            "- 0.0 (Violación de alcance / Datos prohibidos): Solicita credenciales, passwords, tokens privados o temas estrictamente prohibidos.\n"
            "- 0.50 (Desviación menor): Agrega preguntas secundarias o accesorias que no estaban contempladas en `<campos_faltantes>`.\n"
            "- 1.0 (Alcance estricto): Se limita al 100% a indagar los conceptos requeridos sin inventar temas ajenos ni violar la seguridad."
        ),
        criteria=NoulCriteria(
            true="El agente se ciñe de forma quirúrgica a los campos faltantes sin desvíos ni preguntas prohibidas (puntuación 1.0).",
            false="El agente pregunta por temas no solicitados, irrelevantes o prohibidos (puntuación cercana a 0.0).",
        ),
    ),
    "clarification_no_redundancy": Noul(
        instructions=(
            "Evalúa que el agente NUNCA vuelva a solicitar ningún dato que el usuario ya haya provisto "
            "o mencionado en su reporte original (`user_input`) ni temas marcados en `must_not_ask`.\n\n"
            "Guía de calibración de puntuación (0.0 a 1.0):\n"
            "- 0.0 (Redundante): Vuelve a pedir un dato que el usuario ya especificó (ej. código HTTP 500, entorno o navegador ya dados).\n"
            "- 1.0 (Cero redundancia): Cero redundancia; cada pregunta indaga datos verdaderamente nuevos y desconocidos."
        ),
        criteria=NoulCriteria(
            true="Ninguna de las preguntas repite ni solicita información que el usuario ya hubiera mencionado en su reporte (puntuación 1.0).",
            false="El agente vuelve a pedir datos que el usuario ya aportó en su mensaje original (puntuación cercana a 0.0).",
        ),
    ),
    "clarification_rule_compliance": Noul(
        instructions=(
            "Evalúa el cumplimiento riguroso de las reglas de negocio y operativas del nodo de clarificación:\n"
            "1. Cláusula 'No sé' (Importante): Indica amablemente al usuario que si desconoce algún dato puede aclararlo y el ticket se registrará igual.\n"
            "2. Cantidad de preguntas: Formula entre 1 y un máximo de 4 preguntas (3-4 solo cuando el input sea muy vago o complejo).\n"
            "3. Formato de presentación: Si formula 2 o más preguntas, usa lista numerada (1. ... 2. ...). Si es 1 sola pregunta, puede ir en texto fluido o numerada (ninguna de las dos opciones se penaliza).\n"
            "4. Justificación técnica opcional: NO penalizar bajo ninguna circunstancia si no se explica la justificación u objetivo técnico de las preguntas.\n"
            "5. Incidentes de infraestructura/producción: En caídas o timeouts, NO exigir pasos de reproducción manuales ni navegador.\n"
            "6. Idioma: Responde en el mismo idioma en el que está redactado el reporte del usuario.\n\n"
            "Guía de calibración de puntuación (0.0 a 1.0):\n"
            "- 0.0 (Incumplimiento grave): Omite la cláusula anti-frustración ('no sé'), excede las 4 preguntas o viola reglas de formato/infraestructura.\n"
            "- 0.50 (Cumplimiento parcial): Omisión menor de formato pero mantiene la intención general y cordialidad.\n"
            "- 1.0 (Cumplimiento óptimo): Cumple cabalmente todas las reglas de negocio aplicables."
        ),
        criteria=NoulCriteria(
            true="Cumple todas las reglas operativas de formato, cláusula anti-frustración, cantidad de preguntas y adaptabilidad (puntuación 1.0).",
            false="Incumple reglas críticas de formato, omite la cláusula anti-frustración o viola pautas del dominio (puntuación cercana a 0.0).",
        ),
    ),
    "clarification_tone_clarity": Noul(
        instructions=(
            "Evalúa la empatía, el tono profesional, la claridad para un usuario no técnico y el nivel de concisión de la respuesta.\n\n"
            "Guía de calibración de puntuación (0.0 a 1.0):\n"
            "- 0.0 (Tono hostil o salida vacía): Salida vacía, tono robótico, arrogante o incomprensible.\n"
            "- 0.50 (Aceptable pero mejorable): Tono frío, desestructurado o excesivamente largo/verboso.\n"
            "- 1.0 (Óptimo, empático y conciso): Tono profesional, cálido, empático, claro y con la extensión justa."
        ),
        criteria=NoulCriteria(
            true="Tono profesional, empático, claro y con excelente concisión (puntuación 1.0).",
            false="Tono desconsiderado, excesivamente técnico, confuso o salida vacía (puntuación cercana a 0.0).",
        ),
    ),
}

# Alias para compatibilidad hacia atrás
CLARIFICATION_RELEVANCE_QUESTIONS = CLARIFICATION_QUESTIONS

# ------------------------------------------------------------------------------
# F. triage_reasoning (Noul continuo 0.0 .. 1.0)
# ------------------------------------------------------------------------------
TRIAGE_REASONING_QUESTIONS: dict[str, Question] = {
    "triage_reasoning_quality": Noul(
        instructions=(
            "Evalúa la calidad, fundamentación y fidelidad de la justificación técnica (`reasoning`) "
            "proporcionada por el agente de triaje en `duplicate_and_rag_check_node`.\n\n"
            "Criterios de evaluación:\n"
            "1. Fidelidad estricta al contexto histórico (rag_context): Cero alucinación de IDs de tickets, soluciones o estados "
            "que no estén documentados en el contexto recuperado.\n"
            "2. Aplicación correcta de las 4 reglas de negocio de deduplicación:\n"
            "   - Regla 1: Ticket abierto con la misma causa raíz -> Duplicado.\n"
            "   - Regla 2: Ticket cerrado/resuelto con solución o workaround aplicable -> Duplicado.\n"
            "   - Regla 3: Ticket cerrado como rechazado o sin solución técnica documentada -> NO duplicado.\n"
            "   - Regla 4: Regresión explícita tras nuevo deploy/versión -> NO duplicado (mencionando antecedente).\n"
            "   - Regla 5: Falsos positivos por mismo código HTTP o módulo distinto -> NO duplicado.\n"
            "3. Coherencia con el proyecto sugerido (`suggested_project_id`) según el catálogo de proyectos.\n\n"
            "Guía de calibración de puntuación (0.0 a 1.0):\n"
            "- 0.0 (Alucinación o justificación errónea): Inventa tickets que no existen, contradice las reglas de negocio o carece de razonamiento.\n"
            "- 0.50 (Parcial): Razonamiento plausible pero superficial, omite citar el ticket analizado o tiene inconsistencias menores.\n"
            "- 1.0 (Óptimo y fundamentado): Justificación técnica rigurosa basada estrictamente en los hechos del reporte y el contexto RAG, aplicando fielmente las reglas de deduplicación y fundamentando el proyecto seleccionado."
        ),
        criteria=NoulCriteria(
            true="La justificación técnica es sólida, verídica, cita la evidencia del contexto RAG y aplica correctamente las reglas de deduplicación (puntuación 1.0).",
            false="La justificación alucina datos, contradice las reglas de ciclo de vida de tickets o no tiene coherencia técnica (puntuación cercana a 0.0).",
        ),
    ),
}


# ==============================================================================
# 3. Funciones Evaluadoras Compatibles con LangSmith
# ==============================================================================


async def extraction_completeness_evaluator(
    run: Run,
    example: Example | None = None,
) -> EvaluationResult:
    """
    Evaluador LangSmith para la completitud de la extracción técnica (qa_evaluator_node).
    Retorna score 0.0 .. 1.0 basado en Decision Models vía OpenRouter.
    """
    clf = get_decision_evaluator_classifier()
    outputs = run.outputs or {}
    expected = example.outputs if example else {}
    inputs = run.inputs or (example.inputs if example else {})

    eval_state = (
        f"Input del usuario:\n{json.dumps(inputs, ensure_ascii=False, indent=2)}\n\n"
        f"Ground Truth esperado:\n{json.dumps(expected, ensure_ascii=False, indent=2)}\n\n"
        f"Extracción generada por el agente:\n{json.dumps(outputs, ensure_ascii=False, indent=2)}"
    )

    response = await clf.ainvoke(
        {"state": eval_state, "questions": EXTRACTION_COMPLETENESS_QUESTIONS}
    )
    noul_ans = response.nouls.get("extraction_completeness")
    score = noul_ans.noul if noul_ans else 0.0

    return EvaluationResult(
        key="extraction_completeness",
        score=round(float(score), 4),
        comment=f"Score JEV Decision Model: {score}",
    )


async def no_hallucination_evaluator(
    run: Run,
    example: Example | None = None,
) -> EvaluationResult:
    """
    Evaluador LangSmith para ausencia de alucinación técnica en la extracción.
    Retorna score 0.0 .. 1.0 (1.0 = cero alucinación).
    """
    clf = get_decision_evaluator_classifier()
    outputs = run.outputs or {}
    inputs = run.inputs or (example.inputs if example else {})

    eval_state = (
        f"Mensaje original provisto por el usuario:\n{json.dumps(inputs, ensure_ascii=False, indent=2)}\n\n"
        f"Datos técnicos extraídos por el agente:\n{json.dumps(outputs, ensure_ascii=False, indent=2)}"
    )

    response = await clf.ainvoke({"state": eval_state, "questions": NO_HALLUCINATION_QUESTIONS})
    noul_ans = response.nouls.get("no_hallucination")
    score = noul_ans.noul if noul_ans else 0.0

    return EvaluationResult(
        key="no_hallucination",
        score=round(float(score), 4),
        comment=f"Score JEV No-Hallucination: {score}",
    )


async def incident_type_evaluator(
    run: Run,
    example: Example | None = None,
) -> EvaluationResult:
    """
    Evaluador determinista para la categorización del tipo de incidente (qa_evaluator_node).
    Compara la categoría asignada por el agente estrictamente contra el Ground Truth del dataset (Costo 0, Latencia 0).
    """
    outputs = run.outputs or {}
    expected = example.outputs if example else {}

    # Obtener el tipo de incidente generado
    bug_analysis = outputs.get("bug_analysis") if isinstance(outputs, dict) else None
    actual_type = (
        bug_analysis.get("incident_type")
        if isinstance(bug_analysis, dict)
        else getattr(bug_analysis, "incident_type", outputs.get("incident_type"))
    )

    expected_type = expected.get("incident_type") if isinstance(expected, dict) else None

    if not expected_type:
        return EvaluationResult(
            key="incident_type_accuracy",
            score=0.0,
            value=actual_type,
            comment="Fallo: El Ground Truth del dataset no especifica 'incident_type'.",
            extra={"expected": None, "actual": actual_type, "is_match": False},
        )

    is_match = actual_type == expected_type
    score = 1.0 if is_match else 0.0

    return EvaluationResult(
        key="incident_type_accuracy",
        score=score,
        value=actual_type,
        comment=f"Esperado (Ground Truth): '{expected_type}' | Actual: '{actual_type}'",
        extra={
            "expected": expected_type,
            "actual": actual_type,
            "is_match": is_match,
        },
    )


async def sufficiency_decision_evaluator(
    run: Run,
    example: Example | None = None,
) -> EvaluationResult:
    """
    Evaluador determinista (Costo 0) para la decisión de suficiencia (qa_evaluator_node).
    Compara outputs.get("is_sufficient") contra el Ground Truth esperado.
    """
    outputs = run.outputs or {}
    expected = example.outputs if example else {}

    bug_analysis = outputs.get("bug_analysis") if isinstance(outputs, dict) else None
    actual_suff = (
        bug_analysis.get("is_sufficient")
        if isinstance(bug_analysis, dict)
        else getattr(bug_analysis, "is_sufficient", outputs.get("is_sufficient"))
    )

    expected_suff = expected.get("is_sufficient") if isinstance(expected, dict) else None

    if expected_suff is None:
        return EvaluationResult(
            key="sufficiency_accuracy",
            score=0.0,
            value=actual_suff,
            comment="Fallo: El Ground Truth del dataset no especifica 'is_sufficient'.",
            extra={"expected": None, "actual": actual_suff, "is_match": False},
        )

    is_match = bool(actual_suff) == bool(expected_suff)
    score = 1.0 if is_match else 0.0

    return EvaluationResult(
        key="sufficiency_accuracy",
        score=score,
        value=actual_suff,
        comment=f"Esperado (Ground Truth): is_sufficient={expected_suff} | Actual: is_sufficient={actual_suff}",
        extra={
            "expected": expected_suff,
            "actual": actual_suff,
            "is_match": is_match,
        },
    )


# ==============================================================================
# Caché en memoria para optimizar llamadas a OpenRouter en Clarificación
# ==============================================================================
_CLARIFICATION_EVAL_CACHE: dict[str, dict[str, float]] = {}


async def _evaluate_clarification_all_dimensions(
    run: Run,
    example: Example | None = None,
) -> dict[str, float]:
    """
    Evalúa las 5 dimensiones independientes de clarificación en una única llamada a OpenRouter.
    Usa caché por run.id para no repetir invocaciones JEV cuando LangSmith ejecuta cada evaluador.
    """
    cache_key = str(getattr(run, "id", id(run)))
    if cache_key in _CLARIFICATION_EVAL_CACHE:
        return _CLARIFICATION_EVAL_CACHE[cache_key]

    outputs = run.outputs or {}
    inputs = run.inputs or (example.inputs if example else {})
    expected = example.outputs if example else {}

    # Extraer la pregunta generada de forma defensiva
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

    dims = [
        "clarification_coverage",
        "clarification_no_out_of_scope",
        "clarification_no_redundancy",
        "clarification_rule_compliance",
        "clarification_tone_clarity",
    ]

    # Capa 1 determinista (Costo 0): salida vacía -> score 0.0 en todas las dimensiones
    if not question_text or not question_text.strip():
        scores = {dim: 0.0 for dim in dims}
        _CLARIFICATION_EVAL_CACHE[cache_key] = scores
        return scores

    user_input = inputs.get("user_input", "")
    missing_fields = inputs.get("missing_fields", [])
    effective_missing = (
        missing_fields
        if missing_fields
        else ["Contexto adicional, módulo o detalles sobre el comportamiento observado"]
    )
    must_not_ask = (
        expected.get("must_not_ask", expected.get("prohibited_topics", []))
        if isinstance(expected, dict)
        else []
    )

    eval_state = (
        f"Reporte del usuario (user_input):\n{user_input}\n\n"
        f"Campos faltantes requeridos (campos_faltantes):\n"
        f"{json.dumps(effective_missing, ensure_ascii=False, indent=2)}\n\n"
        f"Temas que NO se deben preguntar o ya provistos (must_not_ask):\n"
        f"{json.dumps(must_not_ask, ensure_ascii=False, indent=2)}\n\n"
        f'Pregunta generada por el agente a evaluar:\n"""{question_text}"""'
    )

    clf = get_decision_evaluator_classifier()
    response = await clf.ainvoke({"state": eval_state, "questions": CLARIFICATION_QUESTIONS})

    scores = {}
    for dim in dims:
        noul_ans = response.nouls.get(dim)
        if noul_ans is not None:
            scores[dim] = round(float(getattr(noul_ans, "noul", noul_ans)), 4)
        else:
            # Compatibilidad con fixtures/mocks legados que usaban claves antiguas
            if (
                dim == "clarification_coverage"
                and "clarification_technical_focus" in response.nouls
            ):
                old_ans = response.nouls["clarification_technical_focus"]
                scores[dim] = round(float(getattr(old_ans, "noul", old_ans)), 4)
            elif (
                dim == "clarification_tone_clarity"
                and "clarification_tone_conciseness" in response.nouls
            ):
                old_ans = response.nouls["clarification_tone_conciseness"]
                scores[dim] = round(float(getattr(old_ans, "noul", old_ans)), 4)
            else:
                scores[dim] = 0.0

    _CLARIFICATION_EVAL_CACHE[cache_key] = scores
    return scores


async def clarification_coverage_evaluator(
    run: Run,
    example: Example | None = None,
) -> EvaluationResult:
    """Evaluador LangSmith para la cobertura de campos faltantes (0.0 .. 1.0 al 100%)."""
    scores = await _evaluate_clarification_all_dimensions(run, example)
    score = scores.get("clarification_coverage", 0.0)
    return EvaluationResult(
        key="clarification_coverage",
        score=score,
        comment=f"Score JEV Cobertura de Campos Faltantes: {score}",
        extra=scores,
    )


async def clarification_no_out_of_scope_evaluator(
    run: Run,
    example: Example | None = None,
) -> EvaluationResult:
    """Evaluador LangSmith para la restricción de alcance a missing_fields (0.0 .. 1.0 al 100%)."""
    scores = await _evaluate_clarification_all_dimensions(run, example)
    score = scores.get("clarification_no_out_of_scope", 0.0)
    return EvaluationResult(
        key="clarification_no_out_of_scope",
        score=score,
        comment=f"Score JEV Restricción de Alcance: {score}",
        extra=scores,
    )


async def clarification_no_redundancy_evaluator(
    run: Run,
    example: Example | None = None,
) -> EvaluationResult:
    """Evaluador LangSmith para verificar ausencia de redundancia con el user_input (0.0 .. 1.0 al 100%)."""
    scores = await _evaluate_clarification_all_dimensions(run, example)
    score = scores.get("clarification_no_redundancy", 0.0)
    return EvaluationResult(
        key="clarification_no_redundancy",
        score=score,
        comment=f"Score JEV Cero Redundancia: {score}",
        extra=scores,
    )


async def clarification_rule_compliance_evaluator(
    run: Run,
    example: Example | None = None,
) -> EvaluationResult:
    """Evaluador LangSmith para cumplimiento de reglas operativas ('no sé', max 4 preguntas, formato, etc.) (0.0 .. 1.0 al 100%)."""
    scores = await _evaluate_clarification_all_dimensions(run, example)
    score = scores.get("clarification_rule_compliance", 0.0)
    return EvaluationResult(
        key="clarification_rule_compliance",
        score=score,
        comment=f"Score JEV Cumplimiento de Reglas: {score}",
        extra=scores,
    )


async def clarification_tone_clarity_evaluator(
    run: Run,
    example: Example | None = None,
) -> EvaluationResult:
    """Evaluador LangSmith para empatía, tono profesional, claridad y concisión (0.0 .. 1.0 al 100%)."""
    scores = await _evaluate_clarification_all_dimensions(run, example)
    score = scores.get("clarification_tone_clarity", 0.0)
    return EvaluationResult(
        key="clarification_tone_clarity",
        score=score,
        comment=f"Score JEV Tono y Claridad: {score}",
        extra=scores,
    )


async def clarification_relevance_evaluator(
    run: Run,
    example: Example | None = None,
) -> EvaluationResult:
    """
    Evaluador compuesto/promedio para ask_clarification_node (mantenido para compatibilidad).
    Reporta el promedio simple de las 5 dimensiones independientes.
    """
    scores = await _evaluate_clarification_all_dimensions(run, example)
    vals = list(scores.values())
    avg_score = round(sum(vals) / len(vals), 4) if vals else 0.0
    outputs = run.outputs or {}
    question = outputs.get("question", "") if isinstance(outputs, dict) else str(outputs)
    is_empty = not question or not str(question).strip()

    return EvaluationResult(
        key="clarification_relevance",
        score=avg_score,
        comment=f"Score JEV Clarificación (Promedio de 5 dimensiones): {avg_score}",
        extra={
            "is_empty": is_empty,
            **scores,
            "technical_focus": scores.get("clarification_coverage", 0.0),
            "tone_conciseness": scores.get("clarification_tone_clarity", 0.0),
        },
    )


async def clarification_technical_evaluator(
    run: Run,
    example: Example | None = None,
) -> EvaluationResult:
    """Wrapper compatible hacia atrás para foco técnico (reporta clarification_coverage)."""
    return await clarification_coverage_evaluator(run, example)


async def clarification_tone_evaluator(
    run: Run,
    example: Example | None = None,
) -> EvaluationResult:
    """Wrapper compatible hacia atrás para tono (reporta clarification_tone_clarity)."""
    return await clarification_tone_clarity_evaluator(run, example)


async def duplicate_detection_evaluator(
    run: Run,
    example: Example | None = None,
) -> EvaluationResult:
    """
    Evaluador LangSmith para la detección de duplicados en duplicate_and_rag_check_node.
    Compara is_duplicate y duplicate_issue_id generado contra el Ground Truth esperado.
    """
    outputs = run.outputs or {}
    expected = example.outputs if example else {}

    actual_is_dup = outputs.get("is_duplicate", False)
    expected_is_dup = expected.get("is_duplicate", False)

    actual_dup_id = outputs.get("duplicate_issue_id")
    expected_dup_id = expected.get("duplicate_issue_id")

    # Match de flag booleano
    is_match = actual_is_dup == expected_is_dup

    # Si es duplicado, verificar también si coincide el issue_id
    id_match = True
    if expected_is_dup:
        id_match = (
            (str(actual_dup_id) == str(expected_dup_id))
            if expected_dup_id is not None
            else (actual_dup_id is not None)
        )

    score = 1.0 if (is_match and id_match) else (0.5 if is_match else 0.0)

    return EvaluationResult(
        key="duplicate_detection_accuracy",
        score=score,
        comment=(
            f"Esperado: is_duplicate={expected_is_dup}, issue_id={expected_dup_id} | "
            f"Actual: is_duplicate={actual_is_dup}, issue_id={actual_dup_id}"
        ),
        extra={
            "actual_is_duplicate": actual_is_dup,
            "expected_is_duplicate": expected_is_dup,
            "actual_duplicate_issue_id": actual_dup_id,
            "expected_duplicate_issue_id": expected_dup_id,
        },
    )


async def project_routing_evaluator(
    run: Run,
    example: Example | None = None,
) -> EvaluationResult:
    """
    Evaluador LangSmith para la asignación sugerida de proyecto en duplicate_and_rag_check_node.
    Compara suggested_project_id contra el Ground Truth esperado.
    """
    outputs = run.outputs or {}
    expected = example.outputs if example else {}

    actual_project = str(outputs.get("suggested_project_id", "") or "").strip()
    expected_project = str(expected.get("suggested_project_id", "") or "").strip()

    is_match = (actual_project.lower() == expected_project.lower()) if expected_project else True
    score = 1.0 if is_match else 0.0

    return EvaluationResult(
        key="project_routing_accuracy",
        score=score,
        value=actual_project,
        comment=f"Esperado: '{expected_project}' | Actual: '{actual_project}'",
        extra={
            "actual_suggested_project_id": actual_project,
            "expected_suggested_project_id": expected_project,
        },
    )


async def triage_reasoning_evaluator(
    run: Run,
    example: Example | None = None,
) -> EvaluationResult:
    """
    Evaluador LangSmith para la calidad y fundamentación del razonamiento de triaje (duplicate_and_rag_check_node).
    Retorna score 0.0 .. 1.0 evaluado mediante Decision Models (JEV / OpenRouter).
    """
    outputs = run.outputs or {}
    inputs = run.inputs or (example.inputs if example else {})

    reasoning = outputs.get("reasoning", "")
    if not reasoning or not str(reasoning).strip():
        return EvaluationResult(
            key="triage_reasoning_quality",
            score=0.0,
            comment="Fallo: No se proporcionó ninguna justificación técnica ('reasoning' vacío).",
            extra={"is_empty": True},
        )

    enriched_query = inputs.get("enriched_query", "")
    rag_context = inputs.get("rag_context", "")
    projects_text = inputs.get("projects_text", "")
    is_duplicate = outputs.get("is_duplicate", False)
    duplicate_issue_id = outputs.get("duplicate_issue_id")
    suggested_project_id = outputs.get("suggested_project_id")

    eval_state = (
        f"Reporte actual:\n{enriched_query}\n\n"
        f"Contexto histórico RAG:\n{rag_context}\n\n"
        f"Proyectos disponibles:\n{projects_text}\n\n"
        f"Decisión del agente:\n"
        f"- is_duplicate: {is_duplicate}\n"
        f"- duplicate_issue_id: {duplicate_issue_id}\n"
        f"- suggested_project_id: {suggested_project_id}\n\n"
        f'Justificación técnica (reasoning) a evaluar:\n"""{reasoning}"""'
    )

    clf = get_decision_evaluator_classifier()
    response = await clf.ainvoke({"state": eval_state, "questions": TRIAGE_REASONING_QUESTIONS})
    noul_ans = response.nouls.get("triage_reasoning_quality")
    score = round(float(noul_ans.noul), 4) if noul_ans else 0.0

    return EvaluationResult(
        key="triage_reasoning_quality",
        score=score,
        comment=f"Score JEV Calidad de Razonamiento de Triaje: {score}",
        extra={
            "is_empty": False,
            "reasoning": reasoning,
            "decision_score": score,
        },
    )


__all__ = [
    "OPENROUTER_BASE_URL",
    "DECISION_EVAL_MODEL",
    "get_decision_evaluator_classifier",
    "EXTRACTION_COMPLETENESS_QUESTIONS",
    "NO_HALLUCINATION_QUESTIONS",
    "INCIDENT_TYPE_QUESTIONS",
    "SUFFICIENCY_DECISION_QUESTIONS",
    "CLARIFICATION_QUESTIONS",
    "CLARIFICATION_RELEVANCE_QUESTIONS",
    "TRIAGE_REASONING_QUESTIONS",
    "extraction_completeness_evaluator",
    "no_hallucination_evaluator",
    "incident_type_evaluator",
    "sufficiency_decision_evaluator",
    "clarification_coverage_evaluator",
    "clarification_no_out_of_scope_evaluator",
    "clarification_no_redundancy_evaluator",
    "clarification_rule_compliance_evaluator",
    "clarification_tone_clarity_evaluator",
    "clarification_relevance_evaluator",
    "clarification_technical_evaluator",
    "clarification_tone_evaluator",
    "duplicate_detection_evaluator",
    "project_routing_evaluator",
    "triage_reasoning_evaluator",
]
