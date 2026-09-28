"""
tests/agent_eval/fixtures.py
============================
Dataset estandarizado (ground truth) para evaluar el comportamiento del agente LangGraph:
  1. Solicitud de información y clarificación (qa_evaluator_node, ask_clarification_node, route_qa)
  2. Detección de duplicados y soluciones existentes (duplicate_and_rag_check_node, route_duplicates)
  3. Enrutamiento y creación de tickets (confirm_issue_creation_node, redmine_issue_creator_node)
"""

from __future__ import annotations

from typing import Any

# ==============================================================================
# 1. CASOS DE EVALUACIÓN DE CLARIFICACIÓN / SOLICITUD DE INFORMACIÓN
# ==============================================================================

CLARIFICATION_TEST_CASES: list[dict[str, Any]] = [
    {
        "id": "clarif_vague_01",
        "description": "Reporte extremadamente vago de fallo en checkout.",
        "user_input": "No anda el pago, me tiró un error en la pantalla.",
        "expected_is_sufficient": False,
        "expected_incident_type": ["frontend_ui", "backend_api", "general_bug"],
        "min_questions": 1,
        "expected_route": "ask_clarification",
    },
    {
        "id": "clarif_vague_02",
        "description": "Reporte vago sin pasos ni mensajes de error.",
        "user_input": "Hay un botón que no funciona en el dashboard.",
        "expected_is_sufficient": False,
        "expected_incident_type": ["frontend_ui", "general_bug"],
        "min_questions": 1,
        "expected_route": "ask_clarification",
    },
    {
        "id": "clarif_complete_01",
        "description": "Reporte completo de bug de backend con endpoint, status code, payload y stacktrace.",
        "user_input": (
            "Al hacer POST a /api/v1/checkout con use_shipping_for_billing=true en staging "
            "(navegador Chrome 122 en macOS), la API responde 500 con el siguiente error:\n"
            "KeyError: 'billing_address' en checkout_service.py línea 142.\n"
            "Pasos para reproducir:\n"
            "1. Añadir ítem al carrito cart_991823a.\n"
            "2. En el checkout marcar el checkbox 'Misma dirección'.\n"
            "3. Enviar orden. Falla de inmediato."
        ),
        "expected_is_sufficient": True,
        "expected_incident_type": ["backend_api"],
        "min_questions": 0,
        "expected_route": "duplicate_and_rag_check",
    },
    {
        "id": "clarif_complete_02",
        "description": "Reporte frontend completo con pasos y entorno.",
        "user_input": (
            "En Safari iOS 17.2, al hacer click en 'Confirmar Pago' en el modal de checkout, "
            "el botón queda deshabilitado y no envía la petición HTTP. "
            "Pasos:\n"
            "1. Abrir checkout en iPhone con Safari 17.2.\n"
            "2. Seleccionar tarjeta de crédito.\n"
            "3. Presionar 'Confirmar Pago'. La consola remota muestra excepción silenciosa en requestSubmit."
        ),
        "expected_is_sufficient": True,
        "expected_incident_type": ["frontend_ui"],
        "min_questions": 0,
        "expected_route": "duplicate_and_rag_check",
    },
    {
        "id": "clarif_infra_adaptive_01",
        "description": "Incidente de infraestructura/producción: no debe exigir pasos manuales ni navegador.",
        "user_input": (
            "Caída crítica de la base de datos de producción a las 04:12 UTC. "
            "PostgreSQL dejó de aceptar conexiones y el log arroja "
            "'FATAL: sorry, too many clients already'. pgBouncer muestra el pool saturado."
        ),
        "expected_is_sufficient": True,
        "expected_incident_type": ["infra_outage"],
        "min_questions": 0,
        "expected_route": "duplicate_and_rag_check",
    },
    {
        "id": "clarif_escape_hatch_01",
        "description": "Escape hatch: el usuario no conoce más detalles técnicos tras la pregunta.",
        "user_input": "No anda la exportación.",
        "clarification_response": "No sé qué error da ni tengo logs, solo le di click al botón y se quedó colgado.",
        "clarification_turns": 1,
        "expected_route_after_turn": "duplicate_and_rag_check",
    },
]

# ==============================================================================
# 2. CASOS DE EVALUACIÓN DE DEDUPLICACIÓN Y TRIAJE
# ==============================================================================

DEDUPLICATION_TEST_CASES: list[dict[str, Any]] = [
    {
        "id": "dedup_positive_01",
        "description": "Duplicado exacto de ticket abierto: doble cobro por falta de idempotencia.",
        "user_report": (
            "Detectamos transacciones cobradas por duplicado en el endpoint /api/v1/payments/charge "
            "cuando el usuario presiona dos veces el botón de pago o reintenta por mala conexión móvil."
        ),
        "expected_is_duplicate": True,
        "matching_subject_substring": "Idempotency-Key",
        "expected_route": "respond_existing",
    },
    {
        "id": "dedup_positive_02",
        "description": "Duplicado de incidente de infraestructura: OOMKilled en Celery workers.",
        "user_report": (
            "Los pods de celery-worker-reports en el clúster de Kubernetes se están muriendo con "
            "código OOMKilled 137 cuando los clientes intentan exportar reportes de métricas pesados."
        ),
        "expected_is_duplicate": True,
        "matching_subject_substring": "Celery Worker",
        "expected_route": "respond_existing",
    },
    {
        "id": "dedup_solved_01",
        "description": "Problema con solución conocida documentada en ticket resuelto: error TLS en OpenVPN.",
        "user_report": (
            "No puedo ingresar a la VPN corporativa mediante OpenVPN en Linux. "
            "El log dice 'TLS handshake failed' y 'certificate has expired'."
        ),
        "expected_is_duplicate": True,
        "matching_subject_substring": "OpenVPN",
        "solution_keywords": ["techvanguard-corp-v2.ovpn", "portal", "CA", "chrony"],
        "expected_route": "respond_existing",
    },
    {
        "id": "dedup_false_positive_control_01",
        "description": "Control negativo (NO duplicado): Mismo endpoint /api/v1/checkout pero causa distinta (cupón vs billing_address).",
        "user_report": (
            "Error 400 Bad Request en POST /api/v1/checkout al intentar aplicar un cupón "
            "de descuento que contiene guiones bajos o puntos en el código."
        ),
        "expected_is_duplicate": False,
        "expected_route": "confirm_creation",
        "expected_project_id": "proyecto-prueba",
    },
    {
        "id": "dedup_false_positive_control_02",
        "description": "Control negativo (NO duplicado): Problema de red distinto al de Calico o MTU.",
        "user_report": (
            "El DNS interno CoreDNS en Kubernetes está arrojando timeouts intermitentes "
            "al resolver dominios externos terminados en .com."
        ),
        "expected_is_duplicate": False,
        "expected_route": "confirm_creation",
        "expected_project_id": "infraestructura-de-servicios",
    },
]

# ==============================================================================
# 3. CASOS DE EVALUACIÓN DE ENRUTAMIENTO Y CREACIÓN DE TICKETS
# ==============================================================================

CREATION_AND_ROUTING_TEST_CASES: list[dict[str, Any]] = [
    {
        "id": "route_project_infra",
        "description": "Incidente de infraestructura debe enrutarse a infraestructura-de-servicios.",
        "title_summary": "[Kubernetes] Fallo de Ingress Traefik por saturación de CPU en nodos worker",
        "incident_type": "infra_outage",
        "expected_project_id": "infraestructura-de-servicios",
        "expected_tracker": "Bug",
        "expected_priority": "Alta",
    },
    {
        "id": "route_project_app",
        "description": "Bug de aplicación/API debe enrutarse a proyecto-prueba.",
        "title_summary": "[API Clientes] Error 500 al guardar dirección de entrega con código postal alfanumérico",
        "incident_type": "backend_api",
        "expected_project_id": "proyecto-prueba",
        "expected_tracker": "Bug",
        "expected_priority": "Normal",
    },
    {
        "id": "route_project_frontend",
        "description": "Error visual o de interfaz de usuario debe enrutarse a proyecto-prueba.",
        "title_summary": "[Frontend SPA] Menú desplegable de perfil queda oculto detrás del iframe de mapa",
        "incident_type": "frontend_ui",
        "expected_project_id": "proyecto-prueba",
        "expected_tracker": "Bug",
        "expected_priority": "Normal",
    },
]
