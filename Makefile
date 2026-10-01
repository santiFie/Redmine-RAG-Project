# ==============================================================================
# Makefile - Redmine + LangGraph RAG Project
# ==============================================================================

VENV ?= .venv
PYTHON = $(VENV)/bin/python
LANGGRAPH = $(VENV)/bin/langgraph
PYTEST = $(VENV)/bin/pytest

.PHONY: help install up up-dev up-prod make-up-prod build down docker-up docker-down dev seed-data index index-incremental test test-ragas serve-ragas test-agent-eval mcp-build mcp-run mcp-test clean

help: ## Muestra este mensaje de ayuda
	@echo "Comandos disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Instala dependencias y configura shims de compatibilidad (Ragas/VertexAI)
	$(VENV)/bin/pip install -r requirements.txt
	@$(PYTHON) -c "import site, os; \
	p = os.path.join(site.getsitepackages()[0], 'langchain_community', 'chat_models'); \
	os.makedirs(p, exist_ok=True); \
	f = os.path.join(p, 'vertexai.py'); \
	open(f, 'w').write('try:\n    from langchain_google_vertexai import ChatVertexAI\nexcept ImportError:\n    class ChatVertexAI: pass\n__all__ = [\"ChatVertexAI\"]\n'); \
	print('Shim VertexAI configurado en:', f)"

# ── Modos de Arranque de la Aplicación ───────────────────────────────────────

up-dev: ## Levanta todos los servicios dockerizados de infraestructura menos LangGraph (que se ejecuta en local)
	@echo "Levantando infraestructura Docker (PostgreSQL, Redmine, Qdrant)..."
	@docker compose -f docker-compose.yml up -d
	@echo "Iniciando servidor LangGraph Dev en local..."
	@$(LANGGRAPH) dev --host 127.0.0.1 --port 8123

up-prod: ## Levanta absolutamente todos los servicios dockerizados (langgraph-api, bff, ui, postgres, redmine, qdrant) con rebuild (--build)
	@echo "Levantando todos los servicios en Docker con rebuild forzado (--build)..."
	@docker compose -f docker-compose.yml -f docker-compose.override.yml up --build -d

make-up-prod: up-prod ## Alias de up-prod para levantar todos los servicios dockerizados

dev: up-dev ## Alias de up-dev (modo desarrollo)

up: up-dev ## Alias predeterminado para levantar el entorno de desarrollo

docker-up: ## Levanta la infraestructura de Docker (PostgreSQL, Redmine, Qdrant)
	docker compose up -d

docker-down: ## Detiene la infraestructura de Docker
	docker compose stop

build: ## Reconstruye y levanta todos los contenedores de Docker
	@docker compose -f docker-compose.yml -f docker-compose.override.yml up --build

down: ## Detiene los contenedores de Docker
	@docker compose -f docker-compose.yml -f docker-compose.override.yml down

seed-data: ## Puebla Redmine con tickets técnicos realistas y los indexa en Qdrant/Postgres
	$(PYTHON) src/redmine/generate_realistic_tickets.py --index

index: ## Ejecuta la indexación completa inicial (Redmine → Qdrant)
	$(PYTHON) -c "from src.jobs.indexer_job import run_sync; run_sync()"

index-incremental: ## Indexa solo issues modificados/creados en las últimas 24h
	$(PYTHON) -m src.jobs.indexer_job

test: ## Ejecuta la suite de pruebas con Pytest
	$(PYTEST)

test-agent-eval: ## Ejecuta la evaluación determinística del agente LangGraph (enrutamiento, deduplicación, clarificación)
	$(PYTHON) -m src.evaluation.run_agent_eval --save-report

test-ragas: ## Ejecuta la evaluación RAG con Ragas y levanta el servidor de resultados
	$(PYTEST) -m ragas --log-cli-level=INFO
	@mkdir -p ui/public/results && cp -r tests/ragas/results/* ui/public/results/ 2>/dev/null || true
	@echo "Visualización disponible en http://127.0.0.1:8765/dashboard.html (o en la UI en /tests)"
	cd tests/ragas/results && python3 -m http.server 8765 --bind 127.0.0.1

serve-ragas: ## Inicia únicamente el servidor HTTP de resultados Ragas
	@echo "Visualización disponible en http://127.0.0.1:8765/dashboard.html"
	cd tests/ragas/results && python3 -m http.server 8765 --bind 127.0.0.1

# ── MCP Server (Redmine) ──────────────────────────────────────────────────────
MCP_IMAGE ?= redmine-mcp-server
MCP_DIR    = mcp/redmine

mcp-build: ## Build la imagen Docker del MCP server de Redmine
	docker build -t $(MCP_IMAGE) ./$(MCP_DIR)

mcp-run: mcp-build ## Corre el MCP server de Redmine en modo stdio (interactivo)
	docker run --rm -i \
	  --env-file .env \
	  --network host \
	  $(MCP_IMAGE)

mcp-test: mcp-build ## Smoke test del MCP server (handshake initialize)
	@echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.0.1"}}}' \
	  | docker run --rm -i \
	    --env-file .env \
	    --network host \
	    $(MCP_IMAGE) | python3 -m json.tool

# ── Limpieza ──────────────────────────────────────────────────────────────────
clean: ## Limpia archivos temporales y cachés
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
