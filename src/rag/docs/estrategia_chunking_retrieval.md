# Estrategia de Chunking y Retrieval: Hybrid Search + AutoMerging + AutoRetriever

Este documento describe la arquitectura, funcionamiento, justificación de diseño, beneficios y limitaciones de la estrategia combinada de **Chunking Jerárquico**, **Auto-Retrieval con Filtrado de Metadatos**, **Búsqueda Híbrida (Densa + BM25)** y **AutoMerging (Parent-Child)** implementada en el motor RAG de Redmine.

---

## 1. Arquitectura y Flujo de Funcionamiento

El pipeline opera en dos grandes fases: **Ingesta (Indexación Jerárquica)** y **Recuperación (Multi-Stage Retrieval)**.

```text
============================= FASE DE INGESTA =============================
 [Issue Redmine (JSON)]
         │
         ▼
 [Domain Parsing] ────► Nodo Principal (Asunto + Descripción) + Nodos Comentarios
         │              (Se inyectan metadatos: project, author, status, issue_id)
         ▼
 [HierarchicalNodeParser] (Padres: 1024 tokens | Hijos: 256 tokens)
         ├──────────────────────────────────────────────┐
         ▼                                              ▼
 [Nodos Padre + Hijos]                           [Nodos Hoja / Leaf Nodes]
         │                                              │
         ▼                                              ▼
 [Postgres DocumentStore]                        [Qdrant Vector Store]
 (Jerarquía completa para reconstrucción)       (Vectores densos BGE-M3 + Sparse BM25)


=========================== FASE DE RECUPERACIÓN ===========================
 Consulta del Usuario (ej: "¿Qué bugs cerró Juan en el proyecto Pagos sobre timeouts?")
         │
         ▼
 [1. VectorIndexAutoRetriever (LLM Ingestion)]
    ├── Extrae Query Semántica: "bugs timeouts"
    └── Infiere MetadataFilters: project == "Pagos" AND author == "Juan" AND status == "Cerrada"
         │
         ▼
 [2. Qdrant Hybrid Search (Dense + BM25 con alpha=0.5)]
    └── Aplica filtros de metadatos y busca sobre Nodos Hoja (256 tokens)
         │
         ▼
 [3. AutoMergingRetriever (Threshold ratio = 0.5)]
    ├── Evalúa si los hijos recuperados representan >= 50% de su nodo padre.
    ├── SI: Reemplaza los hijos por el Nodo Padre completo (1024 tokens) desde Postgres.
    └── NO: Mantiene los nodos hoja específicos.
         │
         ▼
 [4. Contexto Final Unificado] ──► Orquestador / LLM de Respuesta
```

---

## 2. Componentes Clave

### A. Chunking Jerárquico (Parent-Child)
- **Nivel Padre (1024 tokens):** Almacenado exclusivamente en `PostgresDocumentStore`. Contiene la narrativa completa del ticket o bloques extensos de contexto.
- **Nivel Hijo (256 tokens):** Nodos hoja indexados en `Qdrant`. Permiten que el embedding capture información semántica granular y precisa sin diluir el vector.

### B. Búsqueda Híbrida (Dense Embeddings + FastEmbed BM25)
- Combina la similitud conceptual de `BAAI/bge-m3` con la coincidencia exacta de términos vía `Qdrant/bm25`.
- Controlada por el parámetro `alpha=0.5` (50% vectorial denso, 50% léxico disperso).

### C. Auto-Retrieval (`VectorIndexAutoRetriever`)
- Utiliza un LLM para inspeccionar la consulta del usuario y el esquema `REDMINE_VECTOR_STORE_INFO`.
- Separa la intención semántica de los criterios duros de filtrado (`issue_id`, `project`, `status`, `author`, `entity_type`), traduciéndolos a filtros nativos en la base de datos vectorial.

### D. AutoMerging (`AutoMergingRetriever`)
- Examina los nodos hoja devueltos. Si suficientes fragmentos hijos pertenecen al mismo documento padre (`simple_ratio_thresh = 0.5`), descarta los fragmentos sueltos y recupera el nodo padre íntegro desde PostgreSQL.

---

## 3. ¿Por qué se tomó esta decisión de diseño?

Los datos de seguimiento de proyectos (Redmine) presentan características particulares que hacen fallar a un RAG tradicional (naive chunking + búsqueda vectorial simple):

1. **Jerga técnica y códigos específicos:** Los tickets contienen identificadores de error (`NullPointerException`, códigos HTTP `502`), nombres de ramas (`fix/auth-jwt`), endpoints (`POST /api/v1/checkout`) y números de ticket (`#4051`). La búsqueda vectorial pura falla con frecuencia al recuperar estos términos literales.
2. **Contexto fragmentado en comentarios:** Comentarios como *"Ya apliqué el fix cambiando el timeout a 30s"* son muy específicos pero carecen de sentido si no se conoce el ticket padre que originó la discusión.
3. **Consultas con filtros implícitos:** Los usuarios no suelen consultar únicamente por concepto, sino acotando por atributos (*"los tickets abiertos de X proyecto"*).

---

## 4. Beneficios con Ejemplos Concretos

| Desafío / Caso de Uso | Comportamiento con esta Arquitectura | Ejemplo Concreto |
| :--- | :--- | :--- |
| **Coincidencias exactas vs. conceptuales** *(Hybrid Search)* | BM25 rescata términos exactos mientras que el vector denso capta sinónimos. | **Query:** *"Error 504 Gateway Timeout en /sync"*<br>BM25 encuentra el log exacto en el ticket `#210`, mientras que dense embeddings rescata tickets que hablen de *"caídas por lentitud en el backend"* aunque no usen el código 504. |
| **Comentarios aislados** *(AutoMerging)* | Si varios comentarios coinciden, se recupera el ticket entero sin perder el hilo. | **Query:** *"¿Cómo se resolvió la lentitud de la base de datos?"*<br>Varios comentarios del ticket `#88` coinciden. En vez de devolver 3 comentarios cortados, el retriever entrega la descripción y el debate completo. |
| **Consultas estructuradas en lenguaje natural** *(AutoRetriever)* | El LLM extrae metadatos sin necesidad de formularios SQL manuales. | **Query:** *"Mostrame las tareas cerradas por María en el proyecto CRM"*<br>AutoRetriever genera `MetadataFilters(project="CRM", author="María", status="Cerrada")` reduciendo el espacio de búsqueda drásticamente. |

---

## 5. Críticas, Trade-offs y Limitaciones

A pesar de su robustez, esta arquitectura introduce compromisos técnicos que deben monitorearse:

### 1. Latencia adicional por inferencia en Retrieval
- **Crítica:** `VectorIndexAutoRetriever` requiere una llamada previa al LLM para inferir los filtros antes de golpear a Qdrant.
- **Impacto:** Añade entre **200 ms y 800 ms** a la fase de búsqueda en comparación con una consulta vectorial directa.
- **Ejemplo:** En preguntas muy genéricas (*"¿Cómo configuro el entorno?"*), la llamada al LLM para verificar filtros es redundante y agrega latencia innecesaria.

### 2. Riesgo de Sobre-filtrado (Over-filtering / Hallucinated Filters)
- **Crítica:** Si el LLM malinterpreta una entidad de la query, puede generar un filtro estricto erróneo que resulte en **cero documentos recuperados**.
- **Ejemplo:** Ante la consulta *"Tickets relacionados con Juan"*, el LLM podría asumir `author == 'Juan'`, omitiendo tickets donde Juan fue asignado, mencionado en un comentario o donde participó como reviewer (`comment_author`).

### 3. Sensibilidad del Umbral de Fusión (`simple_ratio_thresh`)
- **Crítica:** El balance del ratio en `AutoMergingRetriever` es delicado:
  - Si el umbral es **muy alto (ej. 0.8)**: Rara vez se fusionan los nodos, manteniendo contexto fragmentado.
  - Si el umbral es **muy bajo (ej. 0.1)**: Casi cualquier coincidencia menor expande al padre entero, saturando la ventana de contexto del LLM con información irrelevante.

### 4. Complejidad Operativa del Almacenamiento Dual
- **Crítica:** Requiere consistencia transaccional entre dos motores: **Qdrant** (vectores hoja) y **PostgreSQL** (árbol jerárquico).
- **Impacto:** Al actualizar o reindexar un issue (`delete_ref_doc`), se debe garantizar la limpieza en ambos destinos para evitar nodos huérfanos.
