"""
tests/__init__.py
Tests de integración y unitarios del proyecto.

Estructura:
  tests/
  ├── unit/
  │   ├── test_redmine_client.py   — Tests unitarios del cliente Redmine (mock httpx)
  │   ├── test_rag_engine.py       — Tests del motor RAG (mock Qdrant)
  │   └── test_agent_graph.py     — Tests del grafo LangGraph
  └── integration/
      ├── test_redmine_api.py      — Tests contra Redmine real (requiere docker-compose up)
      └── test_rag_pipeline.py     — Tests del pipeline RAG end-to-end
"""
