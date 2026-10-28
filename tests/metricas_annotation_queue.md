# Definición de Métricas para Annotation Queues en LangSmith
> **Cola de Revisión:** `redmine-qa-human-review`  
> **Objetivo:** Facilitar una auditoría humana (HITL) rápida, consistente y libre de fatiga, que permita evaluar la calidad de los nodos y calibrar los modelos automáticos (meta-evaluación).

---

## Estrategia de Evaluación

Para maximizar la agilidad y precisión de la revisión humana, se combinan:
- **Métricas Binarias (`0` / `1`):** Para reglas duras, políticas de seguridad y decisiones booleanas.
- **Escalas Cortas (`0 a 2` o `1 a 3`):** Para graduar calidad técnica, tono y fidelidad sin ambigüedad.
- **Categorías Discretas:** Para evaluar pertinencia contextual o tipos específicos.

---

## 1. `ask_clarification_node` (Preguntas de Clarificación)

Evalúa la calidad, pertinencia, tono y cumplimiento de reglas en las preguntas dirigidas al usuario cuando falta información para el reporte.

| Métrica (`Key` en UI) | Tipo de Dato | Valores / Opciones | Criterio de Evaluación Humana |
| :--- | :--- | :--- | :--- |
| `human_clarification_coverage` | Escala (0 a 2) | • `0`: Insuficiente<br>• `1`: Parcial<br>• `2`: Completa | ¿Preguntó por todos los datos necesarios especificados en `missing_fields` sin omitir ninguno indispensable? |
| `human_no_redundancy_or_prohibited` | Binaria (Boolean) | • `1`: Cumple<br>• `0`: Viola | `0` si repregunta algo ya presente en el input, si pide datos fuera de alcance, o si solicita información prohibida (credenciales, contraseñas, o pasos manuales en caídas de infraestructura). |
| `human_rule_compliance` | Binaria (Boolean) | • `1`: Cumple<br>• `0`: Incumple | ¿Aclara amablemente la opción de responder "no lo sé" (anti-frustración)? ¿Respeta el idioma del usuario y la cantidad máxima de 1 a 4 preguntas? |
| `human_tone_and_conciseness` | Escala (1 a 3) | • `1`: Pobre (robótico, confuso, hostil o muy verboso)<br>• `2`: Aceptable<br>• `3`: Óptimo (empático, cercano y conciso) | Evalúa la claridad de redacción, cortesía profesional y economía de palabras para no abrumar al usuario. |
| `human_evaluator_agreement` | Binaria (Boolean) | • `1`: De acuerdo con JEV / Judge<br>• `0`: En desacuerdo | **Meta-evaluación:** ¿El puntaje automático asignado por Decision Models / LLM Judge fue justo y representativo? Útil para detectar deriva o necesidad de calibración. |

---

## 2. `qa_evaluator_node` (Extracción y Suficiencia)

Evalúa la extracción estructurada del incidente (`BugReportExtraction`), la decisión de frenar o avanzar (`is_sufficient`) y la clasificación del ticket.

| Métrica (`Key` en UI) | Tipo de Dato | Valores / Opciones | Criterio de Evaluación Humana |
| :--- | :--- | :--- | :--- |
| `human_sufficiency_decision` | Binaria (Boolean) | • `1`: Correcto<br>• `0`: Incorrecto | ¿Fue acertada la decisión de `is_sufficient` según la matriz de suficiencia (frenar a pedir aclaración vs. avanzar a triaje/creación)? |
| `human_extraction_fidelity` | Escala (0 a 2) | • `0`: Pobre (omitió datos esenciales o alucinó)<br>• `1`: Parcial (completa pero con imprecisiones menores)<br>• `2`: Exacta (fiel al input y sin alucinación) | Fidelidad técnica y completitud en `technical_details`, `reproduction_steps`, `environment_info` y `title_summary`. |
| `human_triage_taxonomy` | Binaria (Boolean) | • `1`: Correcto<br>• `0`: Incorrecto | ¿Son adecuados el `incident_type` (ej. `backend_api`, `feature_request`, `support_request`), `suggested_tracker` y `suggested_priority`? |
| `human_redmine_markdown_quality` | Escala (1 a 3) | • `1`: Deficiente (desordenado o confuso)<br>• `2`: Aceptable<br>• `3`: Excelente (listo para desarrollo en Redmine) | Legibilidad, organización y utilidad del campo `description_markdown` para el equipo técnico que recibirá el issue. |
| `human_missing_fields_quality` | Categórica | • `aplicable_correcto`<br>• `aplicable_incorrecto`<br>• `no_aplica` | *Solo si `is_sufficient=False`:* ¿Los elementos identificados en `missing_fields` son realmente los que faltan para poder accionar el ticket? |

---

## 3. `duplicate_and_rag_check_node` (Deduplicación y Enrutamiento)

Evalúa el razonamiento frente al contexto histórico de Redmine recuperado vía RAG y la selección del proyecto destino.

| Métrica (`Key` en UI) | Tipo de Dato | Valores / Opciones | Criterio de Evaluación Humana |
| :--- | :--- | :--- | :--- |
| `human_duplicate_verdict` | Escala (0 a 2) | • `0`: Error de decisión<br>• `1`: Flag correcto pero ID erróneo<br>• `2`: Flag e Issue ID exactos | Determina si el veredicto `is_duplicate` y el `duplicate_issue_id` se corresponden fielmente con la evidencia del `rag_context`. |
| `human_project_routing` | Binaria (Boolean) | • `1`: Correcto<br>• `0`: Incorrecto | ¿El `suggested_project_id` asignado corresponde al proyecto/equipo responsable según el catálogo de Redmine? |
| `human_reasoning_groundedness` | Escala (0 a 2) | • `0`: Alucinación o contradicción<br>• `1`: Válido pero superficial<br>• `2`: Sólido y bien fundamentado | Evalúa la justificación técnica en `reasoning`: si explica con claridad las reglas aplicadas (abierto, cerrado con fix, rechazo, regresión) apoyándose en el RAG. |