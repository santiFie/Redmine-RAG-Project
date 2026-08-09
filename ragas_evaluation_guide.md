# Guía Experta de Evaluación RAG con Ragas y LlamaIndex

Este documento explica en detalle cómo está construida la suite de evaluación automatizada para tu RAGEngine utilizando **Ragas** y **LlamaIndex**, basándonos específicamente en los archivos `tests/ragas/test_rag_ragas.py` y `tests/ragas/fixtures.py`.

---

## 1. Conceptos Fundamentales: Las Métricas de Ragas

Ragas (Retrieval Augmented Generation Assessment) es un framework diseñado para evaluar pipelines RAG sin depender exclusivamente de anotaciones humanas masivas, utilizando LLMs como evaluadores (LLM-as-a-Judge). En nuestro código, medimos cuatro dimensiones clave:

### A. Generación (Calidad de la Respuesta)

1. **Faithfulness (Fidelidad):**
   * **¿Qué mide?** Mide si la respuesta generada por el LLM se deriva *única y exclusivamente* de los contextos recuperados. Si el LLM alucina información o añade datos externos correctos pero que no estaban en los documentos provistos, esta métrica baja.
   * **Concepto técnico:** El evaluador (Groq LLM) desglosa la respuesta en afirmaciones (statements) y verifica si cada una puede inferirse del contexto.

2. **Answer Relevancy (Relevancia de la Respuesta):**
   * **¿Qué mide?** Mide qué tan directa y pertinente es la respuesta respecto a la pregunta original del usuario. Penaliza las respuestas evasivas o que dan rodeos innecesarios.
   * **Concepto técnico:** Utiliza tanto el LLM como modelos de embeddings. El LLM intenta generar preguntas inversas a partir de la respuesta dada, y luego compara (vía similitud coseno de los embeddings) si esas preguntas inversas se parecen a la pregunta original.

### B. Recuperación (Calidad del Retriever)

3. **Context Precision (Precisión del Contexto):**
   * **¿Qué mide?** Mide si todos los fragmentos *relevantes* que se recuperaron están posicionados en los primeros lugares (ranking).
   * **Concepto técnico:** Evalúa el orden. Compara los contextos recuperados (`retrieved_contexts`) contra la respuesta correcta o los contextos ideales (`reference_contexts`). Penaliza si el contexto útil aparece en la posición 5 en lugar de la 1.

4. **Context Recall (Recuperación del Contexto):**
   * **¿Qué mide?** Mide si el retriever logró encontrar *toda* la información necesaria para responder la pregunta basándose en el "ground truth" (respuesta de referencia).
   * **Concepto técnico:** El evaluador divide la `reference` (respuesta esperada) en afirmaciones y verifica cuántas de esas afirmaciones pueden ser sustentadas por los contextos recuperados.

---

## 2. Los Datos de Prueba (`fixtures.py`)

Para evaluar un sistema RAG de forma consistente en CI/CD, necesitamos datos sintéticos fijos. Aquí radica la magia del archivo `fixtures.py`.

### `REDMINE_ISSUES_FIXTURE` (El Corpus)
Es una lista de diccionarios que simulan tickets reales de Redmine (con descripciones complejas, IDs, estados y comentarios).
* **Por qué así:** Antes de correr los tests, *este corpus se indexa en una colección temporal de Qdrant*. Esto garantiza que el retriever tenga una base de datos controlada y predecible donde buscar, evitando que los tests dependan de datos de producción cambiantes.

### `RAGAS_TESTSET` (Las Muestras de Evaluación)
Es una lista de diccionarios que forman el "Ground Truth" o verdad absoluta. Cada ítem tiene un propósito técnico estricto para Ragas:

```python
{
    "user_input": "¿Cuál es la causa raíz del error 500 en el endpoint /api/projects?",
    "reference": "La causa raíz... es un NullPointerException...",
    "reference_contexts": [
        "Issue #101: Error 500...",
        "Comentario en Issue #101..."
    ]
}
```

* **`user_input`**: La pregunta que se le enviará al `query_engine`.
* **`reference` (Ground Truth):** La respuesta ideal redactada por un humano (o curada). *¿Para qué se usa?* Ragas la utiliza principalmente para calcular el **Context Recall** (saber qué información *debía* encontrarse) y a veces como apoyo para Context Precision.
* **`reference_contexts`**: Los fragmentos exactos del corpus que justifican la respuesta. *¿Para qué se usa?* Sirve de punto de comparación para evaluar si el retriever recuperó exactamente lo que debía.

**¿Por qué están formulados de esta forma?**
Porque Ragas necesita comparar el triángulo de información: **Input (Pregunta) -> Retrieved (Lo que encontró tu sistema) -> Generated (Lo que respondió tu sistema)** contrastándolo con el **Reference (Lo ideal)**. Las preguntas en el fixture van desde consultas directas ("¿Qué tecnologías se usan?") hasta razonamientos cruzados ("¿Por qué fallaban los tests y cómo se solucionó?"), estresando así todas las métricas.

---

## 3. El Procedimiento de Evaluación (`test_rag_ragas.py`)

El archivo usa `pytest` con un enfoque modular y "fixtures" (contextos) de preparación.

### Paso A: Preparación del Entorno (Fixtures)
1. **`rag_engine`**: Instancia tu clase `RAGEngine` apuntando a `TEST_COLLECTION`. Si la colección no existe, inyecta los `REDMINE_ISSUES_FIXTURE`. Al final de los tests (teardown), borra la colección. *Magia pura de CI/CD: no dejas basura.*
2. **`query_engine`**: LlamaIndex abstrae el retriever y el LLM. Ragas utilizará este objeto para ejecutar internamente las preguntas del TestSet.
3. **Evaluadores (`evaluator_llm`, `evaluator_embeddings`)**: Ragas requiere su propio LLM para "juzgar". Usamos *LlamaIndexLLMWrapper* conectándolo a Groq y HuggingFace para que los jueces tengan las mismas capacidades (o superiores) que el sistema base.
4. **`ragas_dataset`**: Convierte nuestra lista cruda de diccionarios en un objeto `EvaluationDataset` que Ragas puede procesar.

### Paso B: Ejecución de la Evaluación (`evaluate`)
El método principal es `test_ragas_evaluation_runs`. 
Se configura cada métrica para que use nuestro `evaluator_llm` local (Groq), de lo contrario Ragas intentaría usar OpenAI por defecto.

```python
result = evaluate(
    query_engine=query_engine,
    dataset=ragas_dataset,
    metrics=metrics,
)
```
* **Bajo el capó:** La función `evaluate` de la integración con LlamaIndex itera sobre el dataset. Por cada `user_input`, hace una llamada real a tu `query_engine`. Captura la respuesta generada (`response`) y los nodos recuperados (`retrieved_contexts`). Luego pasa todos los datos (generado + referencias) al LLM Evaluador para que asigne las notas matemáticas (de 0.0 a 1.0).

### Paso C: Umbrales y Aserciones (CI/CD Gates)
Los siguientes métodos de prueba (`test_faithfulness_threshold`, etc.) funcionan como "Quality Gates":
* Definen umbrales constantes (`MIN_FAITHFULNESS = 0.5`).
* Hacen un `assert score >= UMBRAL`.
* Si un commit nuevo arruina los prompts o rompe el índice HNSW de Qdrant haciendo que el Context Precision baje de 0.4, **el pipeline de tests fallará**, impidiendo que código degradado llegue a producción.

---

## 4. Tips para ser un Experto en Ragas

1. **Ajuste de Umbrales:** Los umbrales (0.4, 0.5) en tu código son razonables para empezar. En un sistema maduro, querrás subirlos a 0.7 - 0.8. Si una métrica falla constantemente, no bajes el umbral; revisa si el problema es el chunking, el modelo de embedding o el prompt del LLM.
2. **El "Juez" importa:** En tu código usas `llama-3.3-70b-versatile` como evaluador. Es excelente. Para evaluaciones de producción, el modelo juez siempre debe ser igual o más inteligente (más grande) que el modelo que genera la respuesta.
3. **Usa los reportes JSON:** Tu método `_save_results` guarda un `scores.json`. En un entorno experto, ese JSON se envía a LangSmith o a una herramienta de observabilidad para trackear cómo mejoran (o empeoran) las métricas a lo largo de las semanas con diferentes commits.
4. **Iteración de Fixtures:** Cuando los usuarios en producción reporten una respuesta mala, crea un issue sintético en `REDMINE_ISSUES_FIXTURE` y agrega el caso a `RAGAS_TESTSET`. Esto es análogo a escribir un test unitario para un bug encontrado.
