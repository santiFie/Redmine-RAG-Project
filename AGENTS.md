# AGENTS.md

LangGraph agent (Redmine MCP + LlamaIndex RAG). Code, docstrings, and LLM prompts are all in **Spanish** — keep it that way.

**⚠️ REGLA ESTRICTA DE MODIFICACIÓN DE CÓDIGO ⚠️**
**ESTÁ ESTRICTAMENTE PROHIBIDO REALIZAR CUALQUIER MODIFICACIÓN DE CÓDIGO SIN UNA SUGERENCIA PREVIA Y LA APROBACIÓN EXPLÍCITA DEL USUARIO. TODA PROPUESTA DE CAMBIO DEBE SER VALIDADA ANTES DE SU IMPLEMENTACIÓN.**

## Visión General del Orquestador
Este documento describe el diseño del orquestador de servicios implementado con **LangGraph** y **LlamaIndex**. Esta arquitectura está pensada para actuar como el núcleo de la orquestación, gestionando flujos de trabajo complejos, integrando herramientas externas a través de MCP (Model Context Protocol) y recuperando conocimiento estructurado.

## Arquitectura del Grafo (LangGraph)
El orquestador sigue un patrón de enrutamiento condicional basado en intenciones con un estado inmutable gestionado por LangGraph.

### Estado del Agente (`AgentState`)
El estado central mantiene el contexto de la conversación y los resultados intermedios de los nodos [cite: 1]:
*   `messages`: Historial de mensajes gestionado con `add_messages`.
*   `intent`: Clasificación de la intención del usuario (e.g., `query_issue`, `search_docs`).
*   `rag_context` / `redmine_result`: Almacenamiento aislado de los resultados de los sub-grafos.

### Nodos Principales
1.  **`analyze_intent`**: Utiliza inferencia estructurada (`with_structured_output`) para forzar al LLM a clasificar la intención del usuario y dirigir el flujo de trabajo.
2.  **`redmine_agent` (Herramientas MCP)**: Agente ReAct que consume herramientas de Redmine expuestas mediante un servidor MCP en un subproceso aislado (stdio transport).
3.  **`rag_query`**: Delegación a LlamaIndex para la recuperación semántica. Ejecuta la consulta en un hilo separado (`asyncio.to_thread`) para no bloquear el event loop asíncrono.
4.  **`respond`**: Nodo unificado que toma el contexto recuperado (RAG) o de herramientas, construye el prompt final y genera una respuesta cohesiva manteniendo un control estricto sobre el tono.

## Repo boundaries

- This directory is a subfolder of the git repo rooted at `/home/santi/Documentos/LangGraph`. Run `git` commands from that parent; this whole tree is currently untracked.
- Project-local venv: `.venv/` (absolute path is already the project-local one — do not use a parent `.venv`). Use `.venv/bin/python`, `.venv/bin/pytest`, etc.
- No CI, no README. `pyproject.toml` and `requirements.txt` drift apart; `requirements.txt` is the fuller install list. Add new deps to both when they matter.

## LangGraph

- Entrypoint (from `langgraph.json`): `src/agent/graph.py:graph` → graph `redmine_agent`. Dev server: `make dev` / `langgraph dev --port 8123` (needs docker infra up).
- Do **not** compile `graph` with a manual checkpointer — the platform handles Postgres persistence via `POSTGRES_URI`.
- Flow: `analyze_intent` → route → `redmine_agent` (MCP tools) | `rag_query` → `respond` | `respond_general` → END. The `respond` node raises if `rag_context` is empty (search_docs path needs indexed docs).
- `.langgraph_api/` at the project root holds dev-server checkpoint files and is **not** gitignored — never commit it.

## LLM providers — the big gotcha

- `src/agent/graph.py` **hardcodes** providers regardless of config: `get_llm("groq", "openai/gpt-oss-120b", ...)` in `analyze_intent`/`redmine_agent`/`respond`; `get_llm("nvidia", "openai/gpt-oss-120b", ...)` in `respond_general`. They need `GROQ_API_KEY` and `NVIDIA_API_KEY` (neither is in `.env.example`). Changing `LLM_PROVIDER` in `.env` will NOT change the agent.
- `src/utils/config.py` (pydantic settings, openai/anthropic/ollama) is not used by the runtime path — don't refactor assuming it is the source of truth.
- There are **two** registries: `src/utils/get_llm.py` (langchain: ChatGroq/ChatNVIDIA) and `src/rag/utils/get_llm.py` (`_LLM_REGISTRY`, llama_index: Groq/NVIDIA/OpenAILike). The RAG engine uses the llama_index one. Keep them in sync.

>###  Estrategia de Recuperación: Parent-Child (Auto Merging Retriever)
*   **Fragmentación Jerárquica**: Los documentos se procesan utilizando `HierarchicalNodeParser`, generando relaciones entre bloques de mayor tamaño (padres) y menor tamaño (hijos).
*   **Vector Store (Qdrant)**: Almacena e indexa únicamente los nodos hoja para optimizar la exactitud de la búsqueda semántica vectorial.
*   **Document Store (Postgres)**: Mantiene la estructura del árbol jerárquico completo para su reconstrucción.
*   **Flujo de Ejecución**: Cuando una búsqueda semántica coincide con un nodo hijo (por ejemplo, un comentario suelto de un issue), el `AutoMergingRetriever` recupera automáticamente el documento padre completo (el ticket base), garantizando que el orquestador reciba todo el contexto estructural necesario.

## Buenas Prácticas Implementadas
*   **Desacoplamiento de Herramientas (MCP)**: Uso de `langchain-mcp-adapters` para estandarizar el consumo de la API REST de Redmine de forma modular, sin acoplar la lógica de los endpoints directamente en los agentes.
*   **Ejecución Segura LlamaIndex-LangGraph**: Aislamiento de las ejecuciones síncronas de LlamaIndex dentro de la topología asíncrona de LangGraph mediante delegación a worker threads .
*   **Gestión de Configuración Robusta**: Validación en tiempo de arranque mediante variables de entorno tipadas con Pydantic Settings, capturando errores de configuración antes de la ejecución del grafo.

## RAG / infra

- `docker compose up -d` (`make up`/`docker-up`) is required: Postgres 16 (also used by `PostgresDocumentStore` via `POSTGRES_URI`), Qdrant `redmine_docs` collection at :6333, Redmine at :3000.
- Embeddings are always `HuggingFaceInferenceAPIEmbedding` (`HUGGINGFACE_API_KEY`, `EMBED_MODEL`, default `BAAI/bge-m3`) — independent of the `EMBEDDING_PROVIDER` env var.
- Ingestion: `src/jobs/indexer_job.py` (incremental Redmine→Qdrant sync), helpers in `src/redmine/` (`generate_test_tickets.py`, `create_test_ticket.py`).

## MCP server

- `mcp/redmine/` = FastMCP stdio server, packaged in Docker (`make mcp-build` / `mcp-run` / `mcp-test`).
- The graph **bypasses Docker**: it spawns `mcp/redmine/server.py` via `.venv/bin/python` with `PYTHONPATH` set to `mcp/redmine`. This requires `fastmcp` installed in the venv — it's currently NOT installed (only in `mcp/redmine/requirements.txt`), so the redmine branch fails until you `pip install fastmcp`.

## Tests / lint

- `make test` → `.venv/bin/pytest` (`asyncio_mode=auto`, testpaths `tests`). `tests/unit/test_redmine_client.py` is all `pytest.skip` TODOs.
- `tests/ragas/` is an integration suite (marker `ragas`): needs Qdrant + `GROQ_API_KEY` + `HUGGINGFACE_API_KEY`; uses underscore-prefixed metrics (`_Faithfulness`, etc.) deliberately (ragas 0.4 metric-class incompatibility — see the module docstring; don't "fix" it). Creates a temp collection `test_ragas_eval_*` and cleans up. Run: `make test-ragas`, view: `make serve-ragas` → :8765. See `ragas_evaluation_guide.md`.
- Ruff: line-length 100, double quotes, `ignore = ["E501"]`. Run `ruff check .` and `.venv/bin/mypy` after changes (both non-strict-configured in `pyproject.toml`).
