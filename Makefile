# ==============================================================================
# Makefile - Redmine + LangGraph RAG Project
# ==============================================================================

VENV ?= .venv
PYTHON = $(VENV)/bin/python
LANGGRAPH = $(VENV)/bin/langgraph
PYTEST = $(VENV)/bin/pytest

.PHONY: help up down docker-up docker-down dev index index-incremental test test-ragas serve-ragas mcp-build mcp-run mcp-test clean

help: ## Muestra este mensaje de ayuda
	@echo "Comandos disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

docker-up: ## Levanta la infraestructura de Docker (PostgreSQL, Redmine, Qdrant)
	docker compose up -d

docker-down: ## Detiene la infraestructura de Docker
	docker compose down

dev: ## Inicia el servidor de desarrollo de LangGraph
	$(LANGGRAPH) dev --host 127.0.0.1 --port 8123 --allow-blocking

up: docker-up ## Levanta la infraestructura de Docker e inicia LangGraph Dev
	@echo "Servicios Docker iniciados correctamente."
	@echo "Iniciando servidor LangGraph Dev..."
	$(LANGGRAPH) dev --host 127.0.0.1 --port 8123 --allow-blocking

down: docker-down ## Detiene los contenedores de Docker

index: ## Ejecuta la indexación completa inicial (Redmine → Qdrant)
	$(PYTHON) -c "from src.jobs.indexer_job import run_sync; run_sync()"

index-incremental: ## Indexa solo issues modificados/creados en las últimas 24h
	$(PYTHON) -m src.jobs.indexer_job

test: ## Ejecuta la suite de pruebas con Pytest
	$(PYTEST)

test-ragas: ## Ejecuta la evaluación RAG con Ragas y levanta el servidor de resultados
	$(PYTEST) -m ragas
	@echo "Visualización disponible en http://127.0.0.1:8765/dashboard.html"
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
