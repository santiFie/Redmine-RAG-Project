"""
tests/integration/__init__.py
==============================
Tests de integración del proyecto — requieren infraestructura real activa
(docker-compose up -d: Qdrant :6333, Postgres :5432).

Estructura:
  tests/integration/
  └── test_rag_engine_real.py  — Flujo básico de indexación y retrieval con
                                  servicios reales (RAGEngine + Qdrant + Postgres)
"""
