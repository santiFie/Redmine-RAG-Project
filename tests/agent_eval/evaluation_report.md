# Reporte de Evaluación del Agente LangGraph

**Fecha y hora:** 2026-09-08 21:05:39

## Resumen de Métricas

| Subsistema Evaluado | Métrica Principal | Resultado |
| :--- | :--- | :--- |
| Clarificación y Suficiencia | Precisión de Enrutamiento | **100.0%** |
| Deduplicación y Triaje | F1-Score (Equilibrio Precision/Recall) | **100.0%** |
| Deduplicación y Triaje | Control de Falsos Positivos (Precision) | **100.0%** |
| Enrutamiento a Proyectos | Exactitud de Asignación | **100.0%** |

## Detalle de Casos por Subsistema

### Clarificación y Solicitud de Información
| ID | Descripción | Estado |
| :--- | :--- | :--- |
| `clarif_vague_01` | Reporte extremadamente vago de fallo en checkout. | **PASSED** |
| `clarif_vague_02` | Reporte vago sin pasos ni mensajes de error. | **PASSED** |
| `clarif_complete_01` | Reporte completo de bug de backend con endpoint, status code, payload y stacktrace. | **PASSED** |
| `clarif_complete_02` | Reporte frontend completo con pasos y entorno. | **PASSED** |
| `clarif_infra_adaptive_01` | Incidente de infraestructura/producción: no debe exigir pasos manuales ni navegador. | **PASSED** |
| `clarif_escape_hatch_01` | Escape hatch: el usuario no conoce más detalles técnicos tras la pregunta. | **PASSED** |

### Deduplicación y Triaje
| ID | Descripción | Estado |
| :--- | :--- | :--- |
| `dedup_positive_01` | Duplicado exacto de ticket abierto: doble cobro por falta de idempotencia. | **PASSED** |
| `dedup_positive_02` | Duplicado de incidente de infraestructura: OOMKilled en Celery workers. | **PASSED** |
| `dedup_solved_01` | Problema con solución conocida documentada en ticket resuelto: error TLS en OpenVPN. | **PASSED** |
| `dedup_false_positive_control_01` | Control negativo (NO duplicado): Mismo endpoint /api/v1/checkout pero causa distinta (cupón vs billing_address). | **PASSED** |
| `dedup_false_positive_control_02` | Control negativo (NO duplicado): Problema de red distinto al de Calico o MTU. | **PASSED** |

### Enrutamiento a Proyectos y Creación
| ID | Descripción | Estado |
| :--- | :--- | :--- |
| `route_project_infra` | Incidente de infraestructura debe enrutarse a infraestructura-de-servicios. | **PASSED** |
| `route_project_app` | Bug de aplicación/API debe enrutarse a proyecto-prueba. | **PASSED** |
| `route_project_frontend` | Error visual o de interfaz de usuario debe enrutarse a proyecto-prueba. | **PASSED** |

