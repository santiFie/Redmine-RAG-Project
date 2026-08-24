"""
tests/integration/test_rag_engine_real.py
==========================================
Test de integración del RAGEngine con servicios reales.

Objetivo:
  Detectar errores de código en src/rag/engine.py relacionados con la
  indexación, el retrieval y el almacenamiento, usando un único issue de
  ejemplo sin mocks.

Requisitos:
  - docker-compose up -d  (Qdrant :6333, Postgres :5432)
  - Variables de entorno:  POSTGRES_URI, QDRANT_URL, HUGGINGFACE_API_KEY,
                           GROQ_API_KEY (o LLM_PROVIDER + clave correspondiente)

Ejecución:
  .venv/bin/pytest tests/integration/test_rag_engine_real.py -v -m integration
"""

from __future__ import annotations

import logging
import uuid

import pytest

from src.rag.engine import RAGEngine
from tests.ragas.fixtures import REDMINE_ISSUES_FIXTURE

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Issue de ejemplo: #101 — Error 500 en /api/projects
_ISSUE_EJEMPLO = REDMINE_ISSUES_FIXTURE[0]

# Pregunta representativa cuya respuesta debe estar en el issue #101
_PREGUNTA = "¿Cuál es la causa raíz del error 500 en el endpoint /api/projects?"

# ---------------------------------------------------------------------------
# Logging (para seguir el flujo del retrieval durante los tests)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
# Logs propios del motor RAG (incluye nivel DEBUG para ver el auto-merge)
logging.getLogger("src.rag.engine").setLevel(logging.DEBUG)
# Entrañas de LlamaIndex: retrieval, parsing y filtros inferidos por el auto-retriever
logging.getLogger("llama_index.core").setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Fixture principal
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    """
    Instancia un RAGEngine contra una colección Qdrant temporal y carga
    un único issue de ejemplo. Al finalizar el módulo, elimina la colección
    y los nodos del docstore para no dejar basura.
    """
    coleccion_temporal = f"test_engine_real_{uuid.uuid4().hex[:8]}"

    motor = RAGEngine(collection_name=coleccion_temporal)
    motor.index_redmine_issues([_ISSUE_EJEMPLO])

    yield motor

    # Teardown: borrar la colección temporal de Qdrant
    try:
        motor._qdrant.delete_collection(coleccion_temporal)
    except Exception:
        pass

    # Teardown: eliminar los nodos del docstore (Postgres) asociados al issue
    try:
        motor._index.delete_ref_doc(
            str(_ISSUE_EJEMPLO["id"]),
            delete_from_docstore=True,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_indexacion_basica(engine: RAGEngine) -> None:
    """
    Verifica que index_redmine_issues no lanza excepciones y que la
    colección temporal existe en Qdrant después de la indexación.
    """
    assert engine._qdrant.collection_exists(engine.collection_name), (
        f"La colección '{engine.collection_name}' no fue creada en Qdrant "
        "después de indexar el issue de ejemplo."
    )


@pytest.mark.integration
def test_query_retorna_contexto(engine: RAGEngine) -> None:
    """
    Verifica que query() retorna un string no vacío con al menos un
    fragmento relevante al issue #101.
    """
    resultado = engine.query(_PREGUNTA, top_k=3)

    assert isinstance(resultado, str), "query() debe retornar un str."
    assert resultado.strip(), "query() no debe retornar un string vacío."
    assert "Score:" in resultado, (
        "El contexto formateado debe contener la etiqueta 'Score:'."
    )
    assert "Texto:" in resultado, (
        "El contexto formateado debe contener la etiqueta 'Texto:'."
    )


@pytest.mark.integration
async def test_aquery_retorna_contexto(engine: RAGEngine) -> None:
    """
    Variante async de test_query_retorna_contexto.
    Verifica que aquery() retorna el mismo contrato que query() usando el
    AsyncAutoMergingRetriever sin bloquear el event loop.
    """
    resultado = await engine.aquery(_PREGUNTA, top_k=3)

    assert isinstance(resultado, str), "aquery() debe retornar un str."
    assert resultado.strip(), "aquery() no debe retornar un string vacío."
    assert "Score:" in resultado, (
        "El contexto formateado debe contener la etiqueta 'Score:'."
    )
    assert "Texto:" in resultado, (
        "El contexto formateado debe contener la etiqueta 'Texto:'."
    )
