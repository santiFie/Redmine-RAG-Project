# Guion de Demostración Técnica y Pitch para Portfolio (Loom / LinkedIn / Entrevistas)

Este documento contiene el guion estructurado para grabar un **video walkthrough de 2:30 a 3:00 minutos** demostrando el proyecto end-to-end, junto con la publicación de LinkedIn optimizada para reclutadores técnicos y directores de ingeniería de IA.

---

## 🎬 Guion de Grabación de Video (2:30 - 3:00 min)

### Preparación previa a la grabación:
* Levantar el stack completo con `docker compose up -d` y ejecutar `make seed-data`.
* Tener abiertas las pestañas:
  1. **Angular UI** (`http://localhost:4200`): chat limpio listo para interactuar.
  2. **Redmine** (`http://localhost:3000`): listado de issues abierto.
  3. **Resultados de Evaluación** (`http://localhost:4200/tests` o `http://127.0.0.1:8765/dashboard.html`): panel de métricas Ragas.
  4. **GitHub Actions**: visualización de un build en verde del pipeline CI.

---

### Bloque 1: El Gancho y el Problema de Negocio (0:00 - 0:30)
* **Pantalla:** Mostrar brevemente el diagrama de arquitectura del `README.md` y la interfaz del chat en Angular.
* **Qué decir:**
  > *"En empresas de tecnología que gestionan miles de tickets en herramientas como Redmine o Jira, el conocimiento crítico queda enterrado en hilos interminables, y los equipos de soporte pierden incontables horas procesando reportes de bugs incompletos o duplicados.*
  > *Para resolver esto, construí este Copiloto Inteligente de TI y Orquestador de Agentes End-to-End combinando **LangGraph**, **LlamaIndex** y el **Model Context Protocol (FastMCP)**."*

---

### Bloque 2: RAG Jerárquico Avanzado (Parent-Child) (0:30 - 1:10)
* **Pantalla:** En la interfaz de chat de Angular, escribir una consulta técnica de base de conocimiento:
  > *"¿Cómo configuro la conexión VPN y qué certificados necesito según la documentación de soporte?"*
* **Acción:** Enviar mensaje. El agente responde de forma inmediata, estructurada y sin alucinaciones, citando el contexto.
* **Qué decir:**
  > *"En lugar de un chunking plano tradicional que pierde contexto o diluye los embeddings, implementé una arquitectura **Parent-Child con AutoMergingRetriever** de LlamaIndex.*
  > *Indexamos únicamente nodos hoja en **Qdrant** para máxima precisión semántica, mientras que la jerarquía del documento padre vive en **PostgreSQL**. Cuando el usuario consulta, el sistema recupera automáticamente el ticket base completo, logrando un **95.4% de Faithfulness** auditado con Ragas."*

---

### Bloque 3: Flujo Agéntico Multi-Turno & Triaje Inteligente (1:10 - 1:55)
* **Pantalla:** En el chat, simular el reporte de un incidente sin detalles técnicos:
  > *"Tengo un error 500 al intentar finalizar el pago en checkout."*
* **Acción:** El agente detecta que es un `incident_report`, evalúa que falta información crítica (`is_sufficient = False`) y repregunta de forma empática pidiendo pasos para reproducir o logs.
* **Acción usuario:** Responder:
  > *"Ocurre con el endpoint `/api/v1/checkout` con método POST cuando el carrito tiene cupón de descuento, status 500 'Internal Server Error'."*
* **Acción del agente:** El agente ejecuta la deduplicación semántica (`duplicate_and_rag_check`).
  * *Si coincide con un ticket conocido:* Informa que el issue ya está documentado y referencia el ticket existente (ej. #12).
* **Qué decir:**
  > *"Aquí entra en juego la máquina de estados de **LangGraph**. El nodo evaluador analiza la completitud del reporte con esquemas tipados de Pydantic. Si faltan datos, activa un bucle de clarificación adaptativo con escape hatch para evitar bucles infinitos. Luego, busca en la base vectorial si el problema ya fue resuelto históricamente para evitar duplicados."*

---

### Bloque 4: Human-in-the-Loop (HITL) y Mutación con FastMCP (1:55 - 2:30)
* **Pantalla:** Probar un reporte de bug nuevo completo (ej. *"Fallo de renderizado en el modal de confirmación en Safari iOS 17"*).
* **Acción:** El agente detecta que es un bug nuevo y se **pausa** mostrando la solicitud de confirmación:
  > *"Se creará un nuevo ticket para el incidente 'Fallo de renderizado en modal Safari iOS' en el proyecto 'proyecto-prueba'. ¿Confirmas este proyecto o seleccionas otro?"*
* **Acción usuario:** Escribir *"Confirmar"* o seleccionar el botón.
* **Acción agente:** El agente reanuda la ejecución, invoca la herramienta mutativa a través del servidor **FastMCP (stdio transport)** y devuelve el enlace al nuevo ticket generado en Redmine.
* **Pantalla:** Cambiar a la pestaña de Redmine y refrescar: mostrar el nuevo ticket recién creado, con título, prioridad y descripción formateada en Textile/Markdown.
* **Qué decir:**
  > *"Siguiendo el principio de **Separación de Comandos y Consultas (CQS)**, el modelo nunca tiene herramientas de escritura durante el análisis. Una vez validado, usamos la directiva nativa `interrupt()` de LangGraph para poner a un humano en el ciclo antes de crear el ticket. La mutación se ejecuta de forma desacoplada vía **Model Context Protocol** conectándose a la API REST de Redmine."*

---

### Bloque 5: Gobernanza, Aceptación de Feedback y Telemetría (2:30 - 2:50)
* **Pantalla:** En la interfaz de Angular, pasar el cursor sobre la respuesta del agente y hacer click en el botón de feedback (👎 o 👍). Mostrar el modal de categorías de feedback (*Alucinación / Contexto Insuficiente / Enrutamiento Erróneo*) y comentario.
* **Acción:** Enviar feedback de prueba.
* **Qué decir:**
  > *"Para entornos de producción, la gobernanza es fundamental. Cada mensaje cuenta con captura de feedback explícito en el frontend que viaja por el BFF hacia **LangSmith** asociado al `run_id` del LLM. Esto alimenta un **Feedback Flywheel**: los casos reportados con baja puntuación se curan y se convierten sistemáticamente en nuevos casos de prueba en nuestra suite de evaluación determinística, evitando cualquier regresión en futuros despliegues."*

---

### Bloque 6: Evaluación Cuantitativa y Cierre (2:50 - 3:10)
* **Pantalla:** Mostrar brevemente el panel de Ragas (`/tests`) y el pipeline de GitHub Actions en verde.
* **Qué decir:**
  > *"Todo el sistema cuenta con integración continua en GitHub Actions, validación estática estricta con Ruff y MyPy, y suites de evaluación determinística que garantizan un **100% de precisión en enrutamiento** y un control total de falsos positivos.*
  > *El código completo, los informes de decisiones arquitectónicas (ADRs), la especificación de gobernanza y la guía de despliegue en un comando con Docker están disponibles en el repositorio de GitHub. ¡Muchas gracias!"*

---

## 📱 Plantilla de Post para LinkedIn (Alto Alcance Técnico)

Copiar y adaptar el siguiente texto para acompañar el video o capturas en LinkedIn:

```text
🚀 Construí un Agente Autónomo de TI de Grado Enterprise con LangGraph, LlamaIndex y Model Context Protocol (MCP)

La mayoría de los proyectos de IA en portfolio se limitan a scripts sencillos o wrappers básicos de APIs. Quería ir más allá y resolver un problema real de ingeniería: la fragmentación de conocimiento y la sobrecarga operativa en mesas de ayuda y sistemas de tickets como Redmine.

El resultado es un Copiloto Híbrido end-to-end con arquitectura desacoplada:

🔹 Orquestación con LangGraph: Máquina de estados cíclica con clasificación estructurada (Pydantic), ciclos de clarificación multi-turno ante reportes vagos y control estricto de guardrails de entrada y salida.
🔹 Human-in-the-Loop (HITL): Mediante la directiva nativa `interrupt()`, el agente valida con el operador el proyecto y alcance antes de mutar el sistema de tickets.
🔹 RAG Jerárquico (Parent-Child): Superando el chunking plano ingenuo mediante `HierarchicalNodeParser` y `AutoMergingRetriever`. Indexamos únicamente nodos hoja en Qdrant para máxima agudeza semántica, reconstruyendo el ticket padre completo desde PostgreSQL.
🔹 Herramientas con Model Context Protocol (FastMCP): Desacoplamiento total de las APIs externas bajo el principio de Command-Query Separation (CQS), previniendo ataques de prompt injection indirecto.
🔹 Gobernanza de IA & Feedback Flywheel: Captura de feedback explícito (👍/👎 + categorías) en Angular enviado a LangSmith, checkpointing inmutable en PostgreSQL y protocolo de curación para convertir casos fallidos en fixtures de regresión en CI/CD.
🔹 Full-Stack Production-Ready: Frontend reactivo en Angular 18 (SSE streaming), FastAPI BFF con JWT, y pipeline CI/CD en GitHub Actions con Docker Compose.

📊 Hard Evidence & Métricas Reales:
• Ragas RAG Evaluation: 95.4% Faithfulness | 100% Context Recall | 94.0% Context Precision.
• Agent Behavioral Suite: 100% Routing Accuracy | F1-Score 1.0 en deduplicación.

Repo en GitHub con arquitectura completa, ADRs y Quickstart de 1 comando:
👉 [Enlace a tu repositorio]

¿Cómo están abordando la gobernanza, el feedback de los usuarios y el patrón Human-In-The-Loop en sus agentes en producción? Los leo en comentarios. 👇

#AIEngineering #LangGraph #LlamaIndex #GenAI #ModelContextProtocol #FastMCP #AIGovernance #SoftwareArchitecture #Docker #Angular
```

---

## 💼 Respuestas Preparadas para Entrevistas (Método STAR)

### Pregunta: *"¿Por qué usaste LangGraph en lugar de un agente estándar de LangChain o CrewAI?"*
* **Situación:** Requeríamos un flujo determinístico donde ciertas operaciones son de solo lectura y otras mutan un sistema corporativo con impacto real.
* **Tarea:** Evitar alucinaciones en mutaciones de datos y soportar bucles de clarificación con límites estrictos de turnos.
* **Acción:** LangGraph modela el flujo como un autómata de estados finitos (`AgentState`) con checkpointing en PostgreSQL. Esto permite pausar la ejecución de forma nativa con `interrupt()`, aislar herramientas de lectura de las de escritura (CQS), y reintentar nodos de red sin volver a ejecutar inferencias costosas de LLMs.
* **Resultado:** 100% de precisión en enrutamiento y cero mutaciones accidentales en tickets corporativos.

### Pregunta: *"¿Cómo garantizaste que el RAG no sufriera el problema de falta de contexto de chunks pequeños?"*
* **Situación:** En tickets técnicos, un comentario aislado (*"apliqué el parche en producción"*) carece de sentido sin el título del ticket, el stacktrace original y los servicios afectados.
* **Tarea:** Lograr alta precisión en la búsqueda vectorial sin perder la visión holística del ticket.
* **Acción:** Implementé la estrategia Parent-Child con `AutoMergingRetriever` de LlamaIndex. Los comentarios y párrafos (hojas) se vectorizan en Qdrant para el embedding, pero el docstore jerárquico vive en PostgreSQL. Si varios fragmentos del mismo ticket son relevantes, el retriever fusiona automáticamente hacia el documento padre.
* **Resultado:** Evaluado formalmente con Ragas, alcanzamos un **100% de Context Recall** y un **95.4% de Faithfulness**.

### Pregunta: *"¿Cómo gestionas la gobernanza de IA, el feedback del usuario y la mejora continua en producción?"*
* **Situación:** Los modelos de lenguaje en producción sufren deriva de calidad si no se monitoriza la satisfacción del usuario y no se capturan los casos límite (*edge cases*).
* **Tarea:** Construir un sistema auditable que capture feedback cualitativo y cuantitativo para mejorar iterativamente sin generar regresiones.
* **Acción:** Diseñé un ciclo cerrado (*AI Flywheel*): el frontend de Angular permite calificar respuestas (👍/👎) y diagnosticar el motivo del fallo (alucinación, falta de contexto, enrutamiento). El BFF canaliza este evento con el `run_id` hacia LangSmith. Los fallos con `score == 0` se analizan en triaje de IA y se transforman directamente en nuevos casos de prueba dentro de `tests/agent_eval/fixtures.py`.
* **Resultado:** El pipeline de CI/CD en GitHub Actions valida que cualquier ajuste de prompts o actualización de modelos mantenga el 100% de precisión en la suite de pruebas antes de desplegarse, garantizando gobernanza estricta y mejora continua medible.
