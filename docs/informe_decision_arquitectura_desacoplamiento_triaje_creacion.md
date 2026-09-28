# Informe de Decisión Arquitectónica: Desacoplamiento de Detección de Duplicados y Creación de Incidentes

**Fecha:** 2026-09-02  
**Módulo:** `src/agent/graph.py` (Rama Triaje & QA)  
**Estado:** Aprobado / Decidido  

---

## 1. Contexto y Problema Planteado

Durante la implementación del flujo de triaje de incidentes (`incident_report`), se evaluó la conveniencia de consolidar en un **único nodo monolítico** las siguientes responsabilidades:
1. Analizar si el incidente reportado ya está duplicado o resuelto históricamente (vía RAG + LLM).
2. Si está duplicado, informar al usuario la solución o ticket existente.
3. Si NO está duplicado, crear automáticamente el nuevo issue en Redmine e informar su creación.

Actualmente, esta secuencia se encuentra modelada en LangGraph a través de nodos y aristas condicionales desacoplados:
* `duplicate_and_rag_check_node` (evaluación cognitiva y consulta histórica).
* `route_duplicates` (bifurcación condicional basada en estado Pydantic).
* `redmine_issue_creator_node` (ejecución de la acción de creación).
* Nodos de respuesta y salida (`respond_existing_solution_node`, `respond_creation_summary_node`).

El presente informe formaliza el análisis sistémico, de seguridad y de control que fundamenta la decisión de **mantener estos nodos desacoplados**.

---

## 2. Análisis Dimensional

### 2.1. Perspectiva Sistémica y Arquitectónica

* **Principio de Separación de Comandos y Consultas (CQS / SRP):**
  * La detección de duplicados es una operación de **lectura e inferencia cognitiva** pura (idempotente y sin efectos colaterales).
  * La creación del ticket es un **comando con efectos secundarios** mutativos permanentes sobre un sistema externo (Redmine).
  * Unificar ambos pasos acopla innecesariamente la latencia y variabilidad de un LLM con la lógica transaccional de un cliente de API.
* **Resiliencia y Políticas de Reintento (`RetryPolicy`):**
  * Si la llamada a Redmine falla por problemas transitorios de red, timeout o mantenimiento, LangGraph permite aplicar una política de reintento (`RetryPolicy`) acotada exclusivamente al nodo `redmine_creator`.
  * En un nodo unificado, un fallo de conexión forzaría a reejecutar la búsqueda vectorial RAG y la inferencia completa del modelo (aumentando costos de tokens, latencia global y arriesgando una decisión inconsistente en la segunda inferencia).
* **Observabilidad y Checkpointing Granular:**
  * Al contar con nodos aislados, el motor de persistencia (Postgres Checkpointer) y herramientas de telemetría (LangSmith, OpenTelemetry) registran métricas exactas y trazabilidad punto a punto: costo/latencia de la deduplicación vs. latencia y status de la llamada al sistema de tickets.
* **Testabilidad y Mantenimiento:**
  * Facilita el testing unitario independiente mediante mocks simples: evaluar duplicados se valida con fixtures RAG; la creación de tickets se valida verificando los payloads hacia Redmine.

### 2.2. Perspectiva de Seguridad

* **Principio de Mínimo Privilegio en Herramientas (*Least Privilege Tool Access*):**
  * Durante la fase de deduplicación, el LLM únicamente recibe herramientas de solo lectura (`get_issue`, `list_issues`, `list_projects`, `get_project`). **No tiene visibilidad ni acceso a `create_issue`**.
  * Esto previene que una alucinación del modelo o un ataque de **Prompt Injection Indirecto** (incrustado en un ticket histórico recuperado por RAG o en el texto del reporte) provoque la mutación inadvertida de datos.
* **Validación Determinística de Esquemas:**
  * La decisión de avanzar hacia la creación no la toma un LLM llamando herramientas en caliente, sino el grafo ejecutando una condición determinística sobre un esquema tipado Pydantic (`DuplicateCheckResult.is_duplicate`).

### 2.3. Perspectiva de Control y Gobernanza

* **Capacidad de Human-in-the-Loop (HITL):**
  * En entornos corporativos de IT e ITSM, la apertura de tickets con impacto real en colas de trabajo a menudo requiere confirmación del usuario u operador.
  * La separación de nodos permite incorporar de forma natural la función `interrupt()` de LangGraph antes de `redmine_creator` (por ejemplo: *"No se detectaron duplicados. ¿Desea crear el ticket en el proyecto Core con prioridad Normal? [Aprobar/Cancelar]"*). En un nodo unificado, pausar la ejecución a mitad de camino compromete la máquina de estados.
* **Idempotencia y Auditoría de Estado:**
  * El estado global (`State`) refleja hitos inmutables (`is_duplicate`, `duplicate_issue_id`, `created_issue_id`, `created_issue_url`). Permite auditar exactamente por qué y en qué momento se creó o no un incidente.

---

## 3. Decisión y Recomendaciones de Implementación

1. **Mantener la topología desacoplada:**
   * Conservar `duplicate_and_rag_check_node` enfocado exclusivamente en lectura y evaluación de similitud.
   * Conservar la bifurcación condicional `route_duplicates`.
   * Conservar `redmine_issue_creator_node` como un nodo dedicado a la acción mutativa.
2. **Optimización de Nodos de Presentación (Opcional):**
   * Si en el futuro se busca simplificar la cantidad de nodos cosméticos, es viable unificar únicamente los nodos de respuesta final (`respond_existing` y `respond_creation`) en un único formateador de resumen de salida, manteniendo intacto el aislamiento de la lógica de negocio y mutación.
