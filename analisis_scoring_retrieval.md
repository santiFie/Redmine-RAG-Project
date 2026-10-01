# Análisis y Diagnóstico: Comportamiento de Scoring y Retrieval en el Motor RAG

Este documento analiza en profundidad por qué una consulta general sobre un error de conexión a PostgreSQL devolvió el **Ticket #67** con un **score de 1.0000**, a pesar de que el ticket describe un incidente específico de infraestructura (políticas de red Calico en Kubernetes) y no una guía general para resolver el problema reportado.

---

## 1. El Caso de Estudio

### Consulta del Usuario
> *"Estoy obteniendo un error de puerto en Postgresql: "El servidor no escucha El servidor no acepta conexiones: la biblioteca de conexiones informa que no se pudo conectar al servidor: Conexión rechazada ¿Está el servidor ejecutándose en el host "172.24.3.147" y aceptando conexiones TCP/IP en el puerto 5432?" por qué ocurre y como lo puedo solucionar?"*

### Resultado de la Recuperación (Retrieval)
* **Score: 1.0000** → `Issue #67: [Calico] NetworkPolicy bloquea tráfico saliente en puerto 5432 desde el namespace de staging`
* **Score: 0.4612** → `Issue #60: [PostgreSQL] Conexiones agotadas (FATAL: sorry, too many clients already) en el pool de pgBouncer`
* **Score: 0.3170** → `Issue #62: Logs y Métricas (Nginx Ingress / gRPC timeouts)`
* **Score: 0.1345** → `Issue #2: Incidencia Crítica: Caída de la base de datos de producción (PostgreSQL)`
* **Score: 0.0398** → `Issue #1: [Documentación] Guía de Configuración de VPN (OpenVPN)`

---

## 2. Análisis de Causa Raíz

El resultado observado responde a la convergencia de cinco factores:

```
[Consulta del Usuario] (PostgreSQL + Puerto 5432 + Conexión rechazada)
         │
         ├─── 1. Búsqueda Híbrida (BM25 + BGE-M3):
         │       Ticket #67 es el ÚNICO documento con "PostgreSQL" y "5432" por red.
         │       → Queda #1 en BM25 y #1 en Denso.
         │
         ├─── 2. Normalización Min-Max (LlamaIndex Qdrant Fusion):
         │       El #1 denso se escala a 1.0 y el #1 sparse se escala a 1.0.
         │       Fórmula: 0.5 * 1.0 + 0.5 * 1.0 = 1.0000 (Score Inflado Relativo).
         │
         ├─── 3. Chunking (256 tokens):
         │       Ticket #67 tiene ~160 tokens. Entra en una sola hoja sin diluirse.
         │
         ├─── 4. Ausencia de Reranker (Cross-Encoder):
         │       No hay modelo que evalúe si el texto responde a la pregunta.
         │
         └─── 5. Naturaleza del Corpus:
                 La base tiene incidentes internos de Redmine, no manuales de DBA.
```

---

### A. Causa Matemática del Score `1.0000`: Min-Max Score Scaling

En `src/rag/engine.py` la recuperación híbrida utiliza `VectorStoreQueryMode.HYBRID` con `alpha = 0.5`. En la implementación de LlamaIndex para Qdrant (`llama_index/vector_stores/qdrant/utils.py`), la función `relative_score_fusion` aplica una normalización **Min-Max** a los resultados antes de ponderarlos:

1. **Normalización por componente:**
   $$\text{dense\_sim} = \frac{x - \min(\text{dense})}{\max(\text{dense}) - \min(\text{dense})}$$
   $$\text{sparse\_sim} = \frac{x - \min(\text{sparse})}{\max(\text{sparse}) - \min(\text{sparse})}$$

2. **Fusión con $\alpha = 0.5$:**
   $$\text{fused\_sim} = (1 - \alpha) \cdot \text{sparse\_sim} + \alpha \cdot \text{dense\_sim}$$

3. **Propagación en AutoMergingRetriever:**
   Cuando el nodo hoja recuperado se evalúa o se fusiona con su padre, el retriever promedia los scores de los nodos hijos:
   $$\text{score\_padre} = \frac{\sum \text{score\_hijos}}{N} = \frac{1.0000}{1} = 1.0000$$

> **Conclusión técnica:** El score de `1.0000` **no representa una probabilidad ni una similitud coseno absoluta**. Es un valor relativo forzado por la normalización: el documento que queda en primer puesto en ambas ramas (densa y léxica) siempre recibirá matemáticamente un score de `1.0000`.

---

### B. Por qué el Ticket #67 dominó tanto en BM25 como en Denso

* **En BM25 (Sparse):**
  BM25 evalúa la frecuencia de término y la rareza del término en la colección (IDF). Los términos clave de la consulta son **"PostgreSQL"** y **"puerto 5432"**.
  * El Ticket #60 y el Ticket #2 contienen "PostgreSQL", pero tratan sobre saturación de conexiones (`too many clients`), sin mención alguna al puerto 5432.
  * El Ticket #1 habla de VPN (puerto 1194).
  * **El Ticket #67 es el único documento del corpus indexado que combina "PostgreSQL", "puerto 5432", "tráfico" y problemas de conectividad de red.** Obtuvo el score IDF más alto de la base.

* **En Denso (`BAAI/bge-m3`):**
  El modelo de embedding proyecta la semántica de la consulta como: *"incapacidad de establecer conexión socket TCP hacia PostgreSQL en el puerto 5432"*.
  El Ticket #67 describe exactamente: *"todos los pods perdieron visibilidad hacia la base de datos PostgreSQL alojada en el segmento de red (10.0.12.50:5432)"*.
  Semánticamente, un bloqueo de red a nivel de firewall/NetworkPolicy es una causa directa de *"conexión rechazada al puerto 5432"*, logrando la mayor similitud vectorial.

---

### C. Rol del Chunking y Tamaños de Chunks

En `src/rag/utils/parser.py`:
* Nodos padre: **1024 tokens**
* Nodos hoja (hijo): **256 tokens**
* Solapamiento: **30 tokens**

**Diagnóstico sobre el chunking:**
* El Ticket #67 tiene una extensión breve (~160 tokens).
* Al ser menor a 256 tokens, **todo el ticket (asunto, problema, causa, reglas Calico y comandos de verificación) quedó alojado en un único nodo hoja**.
* **Impacto:** No hubo fragmentación ni corte de contexto. La densidad de palabras clave y señales semánticas en ese fragmento fue total, amplificando la coincidencia frente a fragmentos más largos o dispersos.

---

### D. Ausencia de Reranker (Cross-Encoder)

El pipeline actual conecta directamente:
$$\text{AutoRetriever} \longrightarrow \text{Qdrant (Híbrido)} \longrightarrow \text{AutoMergingRetriever} \longrightarrow \text{Prompt LLM}$$

* **Bi-encoders (BGE-M3) y BM25:** Evalúan similitudes de manera independiente y superficial por representación vectorial o conteo léxico. No tienen atención cruzada (*cross-attention*) entre la pregunta y el fragmento.
* **Qué hace un Reranker:** Toma pares `(Pregunta, Documento)` y evalúa si el contenido responde la pregunta formulada.
  * Un Reranker habría identificado que el usuario consulta por un problema estándar de PostgreSQL (`listen_addresses`, servicio detenido o `pg_hba.conf`), mientras que el Ticket #67 trata sobre una actualización puntual de Calico CNI en Kubernetes.
  * Los rerankers devuelven puntuaciones calibradas no normalizadas con Min-Max, evitando el `1.0000` artificial.

---

### E. Desfase de Dominio: Pregunta de DBA vs. Corpus de Redmine

* La pregunta del usuario es una consulta típica de administración de sistemas sobre el mensaje canónico de `psql` (`Connection refused`).
* El corpus de Redmine contiene **historial de tickets e incidencias operativas de la empresa**, no manuales técnicos generales ni guías completas de instalación de bases de datos.
* Al pedirle al RAG que busque en esa colección cerrada, el sistema devolvió el único registro histórico que abordó una pérdida de conectividad en el puerto 5432.

---

## 3. Matriz de Diagnóstico

| Componente | Comportamiento Observado | Impacto en el Score |
| :--- | :--- | :--- |
| **Normalización Min-Max** | Escala el mejor resultado de cada rama a 1.0 | **Crítico**: Genera el valor numérico `1.0000`. |
| **Búsqueda Híbrida (BM25 + Dense)** | Coincidencia en "PostgreSQL" + "5432" | **Alto**: Posiciona al Ticket #67 en el primer lugar en ambos modos. |
| **Chunking (256 tokens)** | El ticket completo (~160 tokens) entra en un nodo | **Medio**: Concentra la señal sin dilución de contexto. |
| **Falta de Reranking** | No se evalúa la relevancia cruzada de la respuesta | **Alto**: Impide penalizar documentos que no responden la pregunta. |
| **Cobertura del Corpus** | No hay guías generales de PostgreSQL | **Medio/Alto**: Limita las opciones disponibles para el retriever. |

---

## 4. Opciones de Mejora y Mitigación (Para evaluar)

Las siguientes alternativas quedan planteadas como propuestas de diseño técnico para su consideración:

1. **Incorporar un nodo de Reranking (Cross-Encoder):**
   * Implementar `bge-reranker-large` o `bge-reranker-v2-m3` como `NodePostprocessor` en LlamaIndex.
   * Beneficio: Reordena y calibra los scores con probabilidades reales de relevancia semántica, penalizando documentos tangenciales.

2. **Revisar la estrategia de fusión de scores:**
   * Evaluar **Reciprocal Rank Fusion (RRF)** o mantener similitudes coseno puras en lugar de Min-Max scaling sobre conjuntos pequeños de resultados.

3. **Guardrails en el prompt de respuesta (`respond-rag`):**
   * Indicar explícitamente al LLM generador que, si los tickets recuperados tratan casos muy particulares de infraestructura (p. ej., Calico) y no responden a la causa raíz general de la consulta, mencione el antecedente interno pero complemente con las verificaciones estándar de PostgreSQL (`systemctl status`, `listen_addresses`, `pg_hba.conf`, firewall del host).

4. **Ampliación de la Base de Conocimiento:**
   * Indexar documentación de arquitectura, runbooks y guías de infraestructura general además de los tickets de incidencias puntuales de Redmine.
