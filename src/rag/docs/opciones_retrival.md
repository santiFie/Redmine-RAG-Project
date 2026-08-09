# Estrategias de Retrieval para Redmine

Para resolver la consulta **"¿Qué tickets están relacionados al Tema A?"** existen tres formas principales de estructurar el pipeline de retrieval:

---

## 1️⃣ Búsqueda Directa por Nodos (Por Defecto)

El *retriever* busca la consulta en la base vectorial y devuelve los **$K$** nodos más cercanos semánticamente.

**Proceso:**
- Si el "Tema A" se menciona en un comentario del ticket `#102` y en la descripción del ticket `#405`, la búsqueda devolverá ambos nodos.

**Resultado:**
- Obtienes los fragmentos específicos que coinciden con la búsqueda. Si el nodo recuperado es solo un comentario corto (por ejemplo *"Ya lo solucioné cambiando el puerto a 8080"*), al LLM le faltará el contexto del ticket original.

---

## 2️⃣ Patrón Parent‑Child (Auto‑Merging Retriever) — La mejor opción para Redmine

Esta estrategia resuelve el problema de la falta de contexto en los comentarios.

```text
[Nodo Padre: Ticket Base #102 (Asunto + Descripción)]
│
├───────────────────────┴───────────────────────┐
▼                                               ▼
[Nodo Hijo 1: Comentario A]                    [Nodo Hijo 2: Comentario B]
   (Búsqueda semántica)                           (Búsqueda semántica)
```

**Proceso:**
- Indexas los comentarios como nodos hijos enlazados al ID del nodo padre (el ticket base).
- La búsqueda semántica consulta sobre los nodos hijos (comentarios y fragmentos pequeños).
- Si un comentario coincide semánticamente con el "Tema A", el retriever recupera automáticamente el documento padre completo (asunto y descripción del ticket).

**Resultado:**
- El orquestador obtiene la imagen completa del ticket sin importar si el término clave apareció en la descripción o en el comentario.

---

## 3️⃣ Búsqueda Jerárquica + Filtrado por Metadatos

En lugar de depender exclusivamente de la similitud vectorial pura, aprovechas los metadatos que guardaste en el parsing estructurado.

### Flujo de trabajo
1. **Consulta del usuario** → Embedder local → Vector de búsqueda
2. **Búsqueda vectorial** → Obtención de los $K$ nodos más relevantes (descripciones o comentarios).
3. **Retriever (Parent‑Child / Metadata Lookup)** →
   - Extrae `metadata["issue_id"]` → IDs candidatos (ej. `[102, 204]`).
   - Reconstruye el contexto uniendo asunto, descripción y comentarios.
4. **LLM / Orquestador** → Recibe el contexto completo reconstruido.
5. **Respuesta final** → Presenta los tickets relacionados.

### Opciones de enriquecimiento en tiempo de ejecución
- **Opción A (100 % RAG):** Consulta la Vector DB por todos los nodos que pertenezcan a los `issue_id` obtenidos y arma el hilo completo.
- **Opción B (Híbrida con MCP):** Usa el RAG sólo para encontrar los `issue_id` candidatos y luego llama al servidor MCP (`GET /issues/{id}.json`) para leer el ticket en su estado vivo desde Redmine.

---

## Resumen

El retrieving se hace sobre ambos (descripciones y comentarios). La clave para que funcione bien es guardar siempre el `issue_id` en los metadatos de cada nodo para poder reconstruir la historia completa del ticket antes de pasársela al LLM.