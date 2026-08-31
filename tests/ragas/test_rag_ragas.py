"""
tests/ragas/test_rag_ragas.py
==============================
Suite de evaluación del RAGEngine con el framework Ragas + integración LlamaIndex.

Métricas evaluadas:
  - Faithfulness          : la respuesta no alucina información ajena al contexto
  - AnswerRelevancy       : la respuesta es pertinente a la pregunta
  - ContextPrecision      : los contextos recuperados relevantes aparecen primero
  - ContextRecall         : el retriever encuentra toda la información necesaria

Umbrales (CI/CD gates):
  - faithfulness      >= 0.5
  - answer_relevancy  >= 0.5
  - context_precision >= 0.4
  - context_recall    >= 0.4

Secciones de tests:
  Base              : testset original de retrocompatibilidad
  A — AutoMerging   : VectorIndexRetriever (DENSE) + AutoMergingRetriever
  B — Hybrid        : VectorIndexAutoRetriever (HYBRID/BM25) + AutoMergingRetriever
  C — Metadata      : mismo retriever que B; preguntas con project/status/author
                      para que VectorIndexAutoRetriever infiera filtros.

Flag --suites (ejecución selectiva):
  Permite elegir qué suites ejecutar sin correr toda la evaluación.
  Valores separados por coma: base, automerging, hybrid, metadata_filter

  Ejemplos:
    .venv/bin/pytest tests/ragas/ -m ragas --suites hybrid
    .venv/bin/pytest tests/ragas/ -m ragas --suites hybrid,automerging
    RAGAS_SUITES=hybrid make test-ragas      # fallback por variable de entorno

  Sin el flag (o con --suites all) se ejecutan todas las suites.
  El skip se aplica en los fixtures de resultado para evitar inicializar
  el RAGEngine y los LLMs innecesariamente.

top_k en retriever:
  - query_engine (base/automerging): top_k=5
  - query_engine_hybrid / query_engine_metadata_filter: top_k=3
    Justificación: con 8 issues en el corpus (~24-40 nodos), top_k=5 recupera
    hasta el 20% del índice, generando ruido que penaliza ContextPrecision.
    La regla top_k ≈ 2–3 × reference_contexts_esperados lleva a top_k=3
    para casos Hybrid y Metadata que tienen 2-3 contexts de referencia.

LLM — FallbackLLM (juez de Ragas y sintetizador del query engine):
  Cascada con timeout 60 s por intento y sin backoff:
    1. Groq       → openai/gpt-oss-120b
    2. NVIDIA     → openai/gpt-oss-120b
    3. OpenRouter → openai/gpt-5.4-nano
  Un error 429 o timeout pasa inmediatamente al siguiente proveedor.
  Logging por llamada: proveedor, modelo, método invocado y tokens consumidos
  (prompt/completion/total), más el acumulado; al desmontar el módulo se emite
  el resumen total de consumo LLM de toda la suite.

Requisitos:
  - docker-compose up -d  (Qdrant :6333, Postgres :5432)
  - Variables de entorno:  POSTGRES_URI, QDRANT_URL,
                           HUGGINGFACE_API_KEY, GROQ_API_KEY

Ejecución:
  .venv/bin/pytest tests/ragas/ -m ragas -v
  make test-ragas

Nota sobre underscore en métricas (_Faithfulness, etc.):
  Se usan las clases internas con prefijo underscore de ragas 0.4 de forma
  deliberada para evitar la incompatibilidad de la API pública de esa versión.
  No "corregir" este patrón.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import logging
import os
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nest_asyncio
import pytest
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.llms import (
    LLM,
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever, VectorIndexRetriever
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from ragas import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from ragas.integrations.llama_index import evaluate
from ragas.llms import LlamaIndexLLMWrapper
from src.rag.utils.get_embedding import get_embedding_model
from ragas.metrics import (
    AnswerRelevancy as _AnswerRelevancy,
)
from ragas.metrics import (
    ContextPrecision as _ContextPrecision,
)
from ragas.metrics import (
    ContextRecall as _ContextRecall,
)
from ragas.metrics import (
    Faithfulness as _Faithfulness,
)
from ragas.run_config import RunConfig

from src.rag.engine import RAGEngine
from tests.ragas.fixtures import (
    RAGAS_TESTSET,
    RAGAS_TESTSET_AUTOMERGING,
    RAGAS_TESTSET_HYBRID,
    RAGAS_TESTSET_METADATA_FILTER,
    REDMINE_ISSUES_FIXTURE,
)

nest_asyncio.apply()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Suites válidas — definición canónica en conftest.py; se importa aquí
# solo para que la fixture suites_activas pueda referenciarla sin importar
# desde conftest (pytest no expone conftest como módulo importable).
# ---------------------------------------------------------------------------

_SUITES_VALIDAS = frozenset({"base", "automerging", "hybrid", "metadata_filter"})


# ---------------------------------------------------------------------------
# Umbrales de calidad (CI/CD gates)
# ---------------------------------------------------------------------------

MIN_FAITHFULNESS: float = 0.5
MIN_ANSWER_RELEVANCY: float = 0.5
MIN_CONTEXT_PRECISION: float = 0.4
MIN_CONTEXT_RECALL: float = 0.4

# Timeout por intento en el FallbackLLM (segundos). Sin backoff entre reintentos.
_TIMEOUT_SEGUNDOS: int = 60

# ---------------------------------------------------------------------------
# Directorio donde se persisten los resultados
# ---------------------------------------------------------------------------

_RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# FallbackLLM — cascada Groq → NVIDIA → OpenRouter para errores 429 / timeout
# ---------------------------------------------------------------------------


def _es_error_429(exc: BaseException) -> bool:
    """
    Determina si la excepción corresponde a un error de rate limit (HTTP 429).

    Detecta:
      - Excepciones cuyo mensaje contiene '429', 'rate limit' o 'too many requests'.
      - httpx.HTTPStatusError y similares que expongan ``.response.status_code``.
    """
    msg = str(exc).lower()
    if any(kw in msg for kw in ("rate limit", "429", "too many requests", "ratelimit")):
        return True
    with contextlib.suppress(AttributeError):
        if exc.response.status_code == 429:  # type: ignore[union-attr]
            return True
    return False


def _extraer_uso_tokens(respuesta: Any) -> tuple[int, int, int]:
    """
    Extrae (prompt_tokens, completion_tokens, total_tokens) de una respuesta LLM.

    Los proveedores compatibles con la API de OpenAI exponen el uso en
    ``respuesta.raw['usage']``; como fallback se revisa
    ``respuesta.additional_kwargs['usage']``.
    """
    uso: dict[str, Any] = {}
    raw = getattr(respuesta, "raw", None)
    if isinstance(raw, dict):
        uso = raw.get("usage") or {}
    if not uso:
        kwargs_adicionales = getattr(respuesta, "additional_kwargs", None)
        if isinstance(kwargs_adicionales, dict):
            uso = kwargs_adicionales.get("usage") or {}
    return (
        int(uso.get("prompt_tokens") or 0),
        int(uso.get("completion_tokens") or 0),
        int(uso.get("total_tokens") or 0),
    )


def _sanitizar_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitiza los kwargs enviados a los LLMs subyacentes.
    Fuerza 'n' <= 1 ya que proveedores como Groq y NVIDIA rechazan
    solicitudes con n > 1 con error HTTP 400 (BadRequestError).
    """
    kw = dict(kwargs)
    if "n" in kw and kw["n"] != 1:
        kw["n"] = 1
    return kw


def _construir_cadena_proveedores() -> list:
    """
    Construye la cadena de proveedores LLM en orden de prioridad.

    Orden:
      1. Groq       → openai/gpt-oss-120b
      2. NVIDIA     → openai/gpt-oss-120b
      3. OpenRouter → openai/gpt-5.4-nano

    Los proveedores cuyo paquete no está instalado se omiten silenciosamente.
    """
    from llama_index.llms.groq import Groq  # noqa: PLC0415

    proveedores: list = [
        Groq(
            model="openai/gpt-oss-120b",
            api_key=os.getenv("GROQ_API_KEY", ""),
            max_retries=1,
        )
    ]

    try:
        from llama_index.llms.nvidia import NVIDIA  # noqa: PLC0415

        proveedores.append(
            NVIDIA(
                model="openai/gpt-oss-120b",
                api_key=os.getenv("NVIDIA_API_KEY", ""),
                max_retries=1,
            )
        )
    except ImportError:
        logger.warning("llama-index-llms-nvidia no instalado — fallback NVIDIA omitido.")

    try:
        from llama_index.llms.openai_like import OpenAILike  # noqa: PLC0415

        proveedores.append(
            OpenAILike(
                model="openai/gpt-5.4-nano",
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                api_base="https://openrouter.ai/api/v1",
                is_chat_model=True,
                max_retries=1,
            )
        )
    except ImportError:
        logger.warning("llama-index-llms-openai-like no instalado — fallback OpenRouter omitido.")

    cadena = " → ".join(
        f"{type(p).__name__}(modelo={getattr(p, 'model', '?')})" for p in proveedores
    )
    logger.info("Cadena de fallback LLM construida: %s", cadena)
    return proveedores


class FallbackLLM(LLM):
    """
    LLM evaluador con fallback en cascada para errores 429 (rate limit).

    Orden de proveedores:
      1. Groq       → openai/gpt-oss-120b
      2. NVIDIA     → openai/gpt-oss-120b
      3. OpenRouter → openai/gpt-5.4-nano

    Por cada llamada se intenta cada proveedor en orden. Si se recibe un error 429
    o un timeout (60 s) se pasa inmediatamente al siguiente sin backoff.
    Si todos los proveedores fallan, se propaga el último error.
    """

    _proveedores: list = PrivateAttr()
    _llamadas_totales: int = PrivateAttr(default=0)
    _tokens_prompt: int = PrivateAttr(default=0)
    _tokens_completion: int = PrivateAttr(default=0)
    _tokens_total: int = PrivateAttr(default=0)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._proveedores = _construir_cadena_proveedores()

    # ------------------------------------------------------------------
    # Observabilidad: registro de éxito y consumo acumulado
    # ------------------------------------------------------------------

    def _registrar_exito(self, proveedor: Any, nombre_metodo: str, respuesta: Any) -> None:
        """
        Registra una llamada LLM exitosa: acumula tokens consumidos y emite
        un log con proveedor, modelo, método invocado y estado del contador.
        """
        prompt_t, completion_t, total_t = _extraer_uso_tokens(respuesta)
        self._llamadas_totales += 1
        self._tokens_prompt += prompt_t
        self._tokens_completion += completion_t
        self._tokens_total += total_t
        logger.info(
            "LLM ok | metodo=%s | proveedor=%s | modelo=%s | "
            "tokens(prompt=%d, completion=%d, total=%d) | acumulado[%s]",
            nombre_metodo,
            type(proveedor).__name__,
            getattr(proveedor, "model", "?"),
            prompt_t,
            completion_t,
            total_t,
            self.resumen_consumo(),
        )

    def resumen_consumo(self) -> str:
        """Resumen legible del consumo acumulado de llamadas y tokens."""
        return (
            f"llamadas={self._llamadas_totales}, "
            f"tokens(prompt={self._tokens_prompt}, "
            f"completion={self._tokens_completion}, total={self._tokens_total})"
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name="fallback-llm",
            context_window=32_768,
            num_output=2_048,
            is_chat_model=True,
        )

    # ------------------------------------------------------------------
    # Helpers internos de fallback
    # ------------------------------------------------------------------

    def _intentar_sync(self, nombre_metodo: str, *args: Any, **kwargs: Any) -> Any:
        """
        Ejecuta ``nombre_metodo`` en cada proveedor con timeout de
        ``_TIMEOUT_SEGUNDOS`` segundos. Sin backoff entre intentos.
        """
        kwargs_limpios = _sanitizar_kwargs(kwargs)
        ultimo_error: BaseException = RuntimeError("No hay proveedores disponibles.")
        for proveedor in self._proveedores:
            try:
                fn = getattr(proveedor, nombre_metodo)
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    futuro = pool.submit(fn, *args, **kwargs_limpios)
                    respuesta = futuro.result(timeout=_TIMEOUT_SEGUNDOS)
                self._registrar_exito(proveedor, nombre_metodo, respuesta)
                return respuesta
            except (concurrent.futures.TimeoutError, TimeoutError) as exc:
                ultimo_error = exc
                logger.warning(
                    "Timeout (%ds) en %s(modelo=%s).%s — pasando al siguiente proveedor.",
                    _TIMEOUT_SEGUNDOS,
                    type(proveedor).__name__,
                    getattr(proveedor, "model", "?"),
                    nombre_metodo,
                )
            except Exception as exc:
                if _es_error_429(exc):
                    ultimo_error = exc
                    logger.warning(
                        "Rate limit (429) en %s(modelo=%s).%s — pasando al siguiente proveedor.",
                        type(proveedor).__name__,
                        getattr(proveedor, "model", "?"),
                        nombre_metodo,
                    )
                else:
                    raise
        logger.error(
            "Todos los proveedores fallaron en '%s'. Último error: %s",
            nombre_metodo,
            ultimo_error,
        )
        raise RuntimeError(
            f"Todos los proveedores fallaron en '{nombre_metodo}'."
        ) from ultimo_error

    async def _intentar_async(self, nombre_metodo: str, *args: Any, **kwargs: Any) -> Any:
        """
        Variante async de ``_intentar_sync``: usa ``asyncio.wait_for`` para el timeout.
        Sin backoff entre intentos.
        """
        kwargs_limpios = _sanitizar_kwargs(kwargs)
        ultimo_error: BaseException = RuntimeError("No hay proveedores disponibles.")
        for proveedor in self._proveedores:
            try:
                coro = getattr(proveedor, nombre_metodo)(*args, **kwargs_limpios)
                respuesta = await asyncio.wait_for(coro, timeout=float(_TIMEOUT_SEGUNDOS))
                self._registrar_exito(proveedor, nombre_metodo, respuesta)
                return respuesta
            except TimeoutError as exc:
                ultimo_error = exc
                logger.warning(
                    "Timeout async (%ds) en %s(modelo=%s).%s — pasando al siguiente proveedor.",
                    _TIMEOUT_SEGUNDOS,
                    type(proveedor).__name__,
                    getattr(proveedor, "model", "?"),
                    nombre_metodo,
                )
            except Exception as exc:
                if _es_error_429(exc):
                    ultimo_error = exc
                    logger.warning(
                        "Rate limit (429) async en %s(modelo=%s).%s — pasando al siguiente.",
                        type(proveedor).__name__,
                        getattr(proveedor, "model", "?"),
                        nombre_metodo,
                    )
                else:
                    raise
        logger.error(
            "Todos los proveedores fallaron (async) en '%s'. Último error: %s",
            nombre_metodo,
            ultimo_error,
        )
        raise RuntimeError(
            f"Todos los proveedores async fallaron en '{nombre_metodo}'."
        ) from ultimo_error

    # ------------------------------------------------------------------
    # Métodos abstractos requeridos por LLM (sync)
    # ------------------------------------------------------------------

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        """Completa el prompt con fallback en cascada."""
        return self._intentar_sync("complete", prompt, formatted=formatted, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """
        Delega a ``complete`` sin streaming real.
        Ragas no usa streaming; este wrapper cumple el contrato de tipo.
        """
        resp = self.complete(prompt, formatted=formatted, **kwargs)
        yield resp

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat con fallback en cascada."""
        return self._intentar_sync("chat", messages, **kwargs)

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        """
        Delega a ``chat`` sin streaming real.
        Ragas no usa streaming; este wrapper cumple el contrato de tipo.
        """
        resp = self.chat(messages, **kwargs)
        yield resp

    # ------------------------------------------------------------------
    # Métodos async (override para usar _intentar_async con asyncio.wait_for)
    # ------------------------------------------------------------------

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Variante async de ``complete`` con fallback y timeout."""
        return await self._intentar_async("acomplete", prompt, formatted=formatted, **kwargs)

    @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Variante async de ``chat`` con fallback y timeout."""
        return await self._intentar_async("achat", messages, **kwargs)

    @llm_chat_callback()
    def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        """
        Delega a ``achat`` sin streaming real.
        Ragas no usa streaming; este stub cumple el contrato de tipos de BaseLLM.
        """

        async def _gen():
            yield await self.achat(messages, **kwargs)

        return _gen()

    @llm_completion_callback()
    def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        """
        Delega a ``acomplete`` sin streaming real.
        Ragas no usa streaming; este stub cumple el contrato de tipos de BaseLLM.
        """

        async def _gen():
            yield await self.acomplete(prompt, formatted=formatted, **kwargs)

        return _gen()


# ---------------------------------------------------------------------------
# Fixture: RAGEngine con colección temporal
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rag_engine():
    """
    Instancia un RAGEngine contra una colección Qdrant temporal y carga
    todos los issues de REDMINE_ISSUES_FIXTURE. En teardown elimina la
    colección y los nodos del docstore para no dejar basura.
    """
    coleccion_temporal = f"test_ragas_eval_{uuid.uuid4().hex[:8]}"

    motor = RAGEngine(collection_name=coleccion_temporal)
    motor.index_redmine_issues(REDMINE_ISSUES_FIXTURE)

    yield motor

    # Teardown: borrar la colección temporal de Qdrant
    with contextlib.suppress(Exception):
        motor._qdrant.delete_collection(coleccion_temporal)

    # Teardown: eliminar nodos del docstore (Postgres) por cada issue indexado
    for issue in REDMINE_ISSUES_FIXTURE:
        with contextlib.suppress(Exception):
            motor._index.delete_ref_doc(
                str(issue["id"]),
                delete_from_docstore=True,
            )


# ---------------------------------------------------------------------------
# Fixtures: QueryEngine base (retrocompatibilidad)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fixtures: QueryEngines por sección
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def suites_activas(request: pytest.FixtureRequest) -> frozenset[str]:
    """
    Devuelve el conjunto de suites habilitadas para esta ejecución.

    Orden de prioridad:
      1. Flag ``--suites`` de pytest (ej. ``--suites hybrid,automerging``)
      2. Variable de entorno ``RAGAS_SUITES`` (ej. ``RAGAS_SUITES=hybrid``)
      3. Sin valor → todas las suites activas

    Valores válidos: ``base``, ``automerging``, ``hybrid``, ``metadata_filter``.
    El valor especial ``all`` activa todas las suites.
    """
    raw = request.config.getoption("--suites") or os.getenv("RAGAS_SUITES") or ""
    if not raw or raw.strip().lower() == "all":
        return _SUITES_VALIDAS
    tokens = {t.strip().lower() for t in raw.split(",") if t.strip()}
    desconocidas = tokens - _SUITES_VALIDAS
    if desconocidas:
        logger.warning(
            "Suites desconocidas ignoradas: %s. Válidas: %s",
            desconocidas,
            _SUITES_VALIDAS,
        )
    activas = tokens & _SUITES_VALIDAS
    if not activas:
        logger.warning(
            "Ninguna suite válida reconocida en --suites='%s'. Activando todas.", raw
        )
        return _SUITES_VALIDAS
    logger.info("Suites Ragas activas: %s", activas)
    return frozenset(activas)


@pytest.fixture(scope="module")
def fallback_llm():
    """
    Instancia única de FallbackLLM compartida por:
      - el sintetizador de respuestas de los cuatro QueryEngines, y
      - el LLM juez que usa Ragas para evaluar las métricas.

    Al desmontar el módulo emite en el log el resumen del consumo total
    de llamadas y tokens de toda la suite.
    """
    llm = FallbackLLM()
    yield llm
    logger.info("Consumo total LLM del módulo ragas: %s", llm.resumen_consumo())


# ---------------------------------------------------------------------------
# Fixtures: QueryEngines por sección (sintetizan con FallbackLLM compartido)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def query_engine(fallback_llm: FallbackLLM, rag_engine: RAGEngine):
    retriever = rag_engine._build_base_retriever(top_k=3)
    merging_retriever = AutoMergingRetriever(
        vector_retriever=retriever,
        storage_context=rag_engine._storage_context,
        simple_ratio_thresh=0.5,
        verbose=False,
    )
    return RetrieverQueryEngine.from_args(merging_retriever, llm=fallback_llm)


@pytest.fixture(scope="module")
def query_engine_automerging(fallback_llm: FallbackLLM, rag_engine: RAGEngine):
    base_retriever = VectorIndexRetriever(
        index=rag_engine._index,
        similarity_top_k=5,
        vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
    )
    merging_retriever = AutoMergingRetriever(
        vector_retriever=base_retriever,
        storage_context=rag_engine._storage_context,
        simple_ratio_thresh=0.5,
        verbose=False,
    )
    return RetrieverQueryEngine.from_args(merging_retriever, llm=fallback_llm)


@pytest.fixture(scope="module")
def query_engine_hybrid(fallback_llm: FallbackLLM, rag_engine: RAGEngine):
    base_retriever = rag_engine._build_base_retriever(top_k=3)
    merging_retriever = AutoMergingRetriever(
        vector_retriever=base_retriever,
        storage_context=rag_engine._storage_context,
        simple_ratio_thresh=0.5,
        verbose=False,
    )
    return RetrieverQueryEngine.from_args(merging_retriever, llm=fallback_llm)


@pytest.fixture(scope="module")
def query_engine_metadata_filter(fallback_llm: FallbackLLM, rag_engine: RAGEngine):
    base_retriever = rag_engine._build_base_retriever(top_k=3)
    merging_retriever = AutoMergingRetriever(
        vector_retriever=base_retriever,
        storage_context=rag_engine._storage_context,
        simple_ratio_thresh=0.5,
        verbose=False,
    )
    return RetrieverQueryEngine.from_args(merging_retriever, llm=fallback_llm)


# ---------------------------------------------------------------------------
# Fixtures: LLM y Embeddings evaluadores (usados por Ragas como jueces)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def evaluator_llm(fallback_llm: FallbackLLM):
    """
    LLM evaluador (juez de Ragas): wrapper sobre la misma instancia compartida
    de FallbackLLM que usan los query engines, de modo que el consumo de
    tokens queda centralizado en un único contador.
    """
    return LlamaIndexLLMWrapper(fallback_llm)


@pytest.fixture(scope="module")
def evaluator_embeddings():
    """
    Modelo de embeddings envuelto en el wrapper de Ragas.
    Usa el mismo proveedor configurable que el RAGEngine (local por defecto).
    """
    embed = get_embedding_model()
    return LlamaIndexEmbeddingsWrapper(embed)


# ---------------------------------------------------------------------------
# Helpers: construcción de datasets y persistencia de resultados
# ---------------------------------------------------------------------------


def _construir_dataset(testset: list[dict]) -> EvaluationDataset:
    """
    Convierte una lista del testset en un EvaluationDataset de Ragas.
    Ignora claves extra (bm25_hint, metadata_filter) que no consume Ragas.
    """
    muestras = [
        SingleTurnSample(
            user_input=entrada["user_input"],
            reference=entrada["reference"],
            reference_contexts=entrada["reference_contexts"],
        )
        for entrada in testset
    ]
    return EvaluationDataset(samples=muestras)


def _save_results(resultado: Any, suffix: str = "") -> None:
    """
    Guarda los scores de la evaluación en un archivo JSON dentro de
    tests/ragas/results/ con el timestamp de ejecución en el nombre.

    Args:
        resultado: Objeto de resultado de Ragas con método ``to_pandas()``.
        suffix:    Identificador de la sección (ej. "automerging", "hybrid").
                   Si se omite, el archivo se llama ``scores_<timestamp>.json``.

    No lanza excepciones — un fallo de escritura no debe romper los tests.
    """
    try:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        nombre = f"scores_{suffix}_{timestamp}.json" if suffix else f"scores_{timestamp}.json"
        archivo = _RESULTS_DIR / nombre

        scores = (
            resultado.to_pandas()[
                ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            ]
            .mean()
            .to_dict()
        )
        scores["timestamp"] = timestamp
        if suffix:
            scores["suite"] = suffix

        archivo.write_text(json.dumps(scores, indent=2, ensure_ascii=False))

        # Archivo canónico (sobreescrito en cada run): permite que el dashboard
        # encuentre siempre el resultado más reciente sin listar el directorio.
        nombre_canonico = f"scores_{suffix}_latest.json" if suffix else "scores_latest.json"
        (_RESULTS_DIR / nombre_canonico).write_text(
            json.dumps(scores, indent=2, ensure_ascii=False)
        )
    except Exception:
        pass  # El guardado es best-effort; no interrumpir el test


# ---------------------------------------------------------------------------
# Fixtures: Datasets de Ragas por sección
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ragas_dataset():
    """Dataset base de Ragas (retrocompatibilidad con RAGAS_TESTSET original)."""
    return _construir_dataset(RAGAS_TESTSET)


@pytest.fixture(scope="module")
def ragas_dataset_automerging():
    """Dataset de Ragas para la sección AutoMerging (preguntas padre↔hijo)."""
    return _construir_dataset(RAGAS_TESTSET_AUTOMERGING)


@pytest.fixture(scope="module")
def ragas_dataset_hybrid():
    """Dataset de Ragas para la sección Hybrid (terminología léxica BM25)."""
    return _construir_dataset(RAGAS_TESTSET_HYBRID)


@pytest.fixture(scope="module")
def ragas_dataset_metadata_filter():
    """Dataset de Ragas para la sección Metadata Filtering (project/status/author)."""
    return _construir_dataset(RAGAS_TESTSET_METADATA_FILTER)


# ---------------------------------------------------------------------------
# Fixtures: Resultados de evaluación cacheados por sección
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ragas_result(
    suites_activas: frozenset[str],
    query_engine: Any,
    ragas_dataset: EvaluationDataset,
    evaluator_llm: LlamaIndexLLMWrapper,
    evaluator_embeddings: LlamaIndexEmbeddingsWrapper,
):
    """
    Ejecuta la evaluación Ragas base UNA sola vez (retrocompatibilidad).
    Persiste scores en results/scores_<timestamp>.json.

    Se omite si 'base' no está en las suites activas (flag --suites).
    """
    if "base" not in suites_activas:
        pytest.skip("Suite 'base' no seleccionada en --suites.")
    metricas = [
        _Faithfulness(llm=evaluator_llm),
        _AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings, strictness=1),
        _ContextPrecision(llm=evaluator_llm),
        _ContextRecall(llm=evaluator_llm),
    ]
    resultado = evaluate(
        query_engine=query_engine,
        dataset=ragas_dataset,
        metrics=metricas,
        run_config=RunConfig(max_workers=3, timeout=120),
    )
    _save_results(resultado)
    return resultado


@pytest.fixture(scope="module")
def ragas_result_automerging(
    suites_activas: frozenset[str],
    query_engine_automerging: Any,
    ragas_dataset_automerging: EvaluationDataset,
    evaluator_llm: LlamaIndexLLMWrapper,
    evaluator_embeddings: LlamaIndexEmbeddingsWrapper,
):
    """
    Evaluación Ragas para la sección AutoMerging (DENSE + merge padre↔hijo).
    Persiste scores en results/scores_automerging_<timestamp>.json.

    Se omite si 'automerging' no está en las suites activas (flag --suites).
    """
    if "automerging" not in suites_activas:
        pytest.skip("Suite 'automerging' no seleccionada en --suites.")
    metricas = [
        _Faithfulness(llm=evaluator_llm),
        _AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings, strictness=1),
        _ContextPrecision(llm=evaluator_llm),
        _ContextRecall(llm=evaluator_llm),
    ]
    resultado = evaluate(
        query_engine=query_engine_automerging,
        dataset=ragas_dataset_automerging,
        metrics=metricas,
        run_config=RunConfig(max_workers=3, timeout=120),
    )
    _save_results(resultado, suffix="automerging")
    return resultado


@pytest.fixture(scope="module")
def ragas_result_hybrid(
    suites_activas: frozenset[str],
    query_engine_hybrid: Any,
    ragas_dataset_hybrid: EvaluationDataset,
    evaluator_llm: LlamaIndexLLMWrapper,
    evaluator_embeddings: LlamaIndexEmbeddingsWrapper,
):
    """
    Evaluación Ragas para la sección Hybrid (BM25 + dense + AutoMerging).
    Persiste scores en results/scores_hybrid_<timestamp>.json.

    Se omite si 'hybrid' no está en las suites activas (flag --suites).
    """
    if "hybrid" not in suites_activas:
        pytest.skip("Suite 'hybrid' no seleccionada en --suites.")
    metricas = [
        _Faithfulness(llm=evaluator_llm),
        _AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings, strictness=1),
        _ContextPrecision(llm=evaluator_llm),
        _ContextRecall(llm=evaluator_llm),
    ]
    resultado = evaluate(
        query_engine=query_engine_hybrid,
        dataset=ragas_dataset_hybrid,
        metrics=metricas,
        run_config=RunConfig(max_workers=3, timeout=120),
    )
    _save_results(resultado, suffix="hybrid")
    return resultado


@pytest.fixture(scope="module")
def ragas_result_metadata_filter(
    suites_activas: frozenset[str],
    query_engine_metadata_filter: Any,
    ragas_dataset_metadata_filter: EvaluationDataset,
    evaluator_llm: LlamaIndexLLMWrapper,
    evaluator_embeddings: LlamaIndexEmbeddingsWrapper,
):
    """
    Evaluación Ragas para la sección Metadata Filtering (filtros inferidos del texto).
    Persiste scores en results/scores_metadata_filter_<timestamp>.json.

    Se omite si 'metadata_filter' no está en las suites activas (flag --suites).
    """
    if "metadata_filter" not in suites_activas:
        pytest.skip("Suite 'metadata_filter' no seleccionada en --suites.")
    metricas = [
        _Faithfulness(llm=evaluator_llm),
        _AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings, strictness=1),
        _ContextPrecision(llm=evaluator_llm),
        _ContextRecall(llm=evaluator_llm),
    ]
    resultado = evaluate(
        query_engine=query_engine_metadata_filter,
        dataset=ragas_dataset_metadata_filter,
        metrics=metricas,
        run_config=RunConfig(max_workers=3, timeout=120),
    )
    _save_results(resultado, suffix="metadata_filter")
    return resultado


# ---------------------------------------------------------------------------
# Tests base (retrocompatibilidad)
# ---------------------------------------------------------------------------


@pytest.mark.ragas
@pytest.mark.integration
def test_ragas_evaluation_runs(ragas_result: Any) -> None:
    """
    Verifica que la evaluación se ejecuta correctamente y retorna un
    objeto con las cuatro métricas esperadas sin errores.
    """
    df = ragas_result.to_pandas()

    assert not df.empty, "El resultado de la evaluación Ragas no debe estar vacío."
    for metrica in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        assert metrica in df.columns, (
            f"La métrica '{metrica}' no está presente en el resultado de Ragas."
        )


@pytest.mark.ragas
@pytest.mark.integration
def test_faithfulness_threshold(ragas_result: Any) -> None:
    """
    Gate de calidad: Faithfulness promedio debe ser >= MIN_FAITHFULNESS.

    Faithfulness mide que la respuesta no alucine información ajena al contexto.
    """
    score = ragas_result.to_pandas()["faithfulness"].mean()
    assert score >= MIN_FAITHFULNESS, (
        f"Faithfulness ({score:.3f}) por debajo del umbral mínimo ({MIN_FAITHFULNESS}). "
        "Revisar el prompt del nodo 'respond' o la calidad del retriever."
    )


@pytest.mark.ragas
@pytest.mark.integration
def test_answer_relevancy_threshold(ragas_result: Any) -> None:
    """
    Gate de calidad: AnswerRelevancy promedio debe ser >= MIN_ANSWER_RELEVANCY.

    AnswerRelevancy mide qué tan pertinente es la respuesta respecto a la pregunta.
    """
    score = ragas_result.to_pandas()["answer_relevancy"].mean()
    assert score >= MIN_ANSWER_RELEVANCY, (
        f"AnswerRelevancy ({score:.3f}) por debajo del umbral mínimo ({MIN_ANSWER_RELEVANCY}). "
        "Revisar el chunking o el prompt de generación de respuesta."
    )


@pytest.mark.ragas
@pytest.mark.integration
def test_context_precision_threshold(ragas_result: Any) -> None:
    """
    Gate de calidad: ContextPrecision promedio debe ser >= MIN_CONTEXT_PRECISION.

    ContextPrecision mide si los fragmentos relevantes aparecen en las
    primeras posiciones del ranking del retriever.
    """
    score = ragas_result.to_pandas()["context_precision"].mean()
    assert score >= MIN_CONTEXT_PRECISION, (
        f"ContextPrecision ({score:.3f}) por debajo del umbral mínimo ({MIN_CONTEXT_PRECISION}). "
        "Revisar la configuración del índice HNSW en Qdrant o el parámetro alpha del retriever híbrido."
    )


@pytest.mark.ragas
@pytest.mark.integration
def test_context_recall_threshold(ragas_result: Any) -> None:
    """
    Gate de calidad: ContextRecall promedio debe ser >= MIN_CONTEXT_RECALL.

    ContextRecall mide si el retriever recupera toda la información necesaria
    para responder la pregunta según el ground truth.
    """
    score = ragas_result.to_pandas()["context_recall"].mean()
    assert score >= MIN_CONTEXT_RECALL, (
        f"ContextRecall ({score:.3f}) por debajo del umbral mínimo ({MIN_CONTEXT_RECALL}). "
        "Revisar el tamaño de chunks, el top_k del retriever o la estrategia AutoMerging."
    )


# ---------------------------------------------------------------------------
# Sección A — AutoMergingRetrieval
# Retriever: VectorIndexRetriever (DENSE) + AutoMergingRetriever
# Casos: preguntas que requieren merge del nodo hijo (comentario) con el
#        nodo padre (descripción), ejercitando el ratio threshold de merge.
# ---------------------------------------------------------------------------


@pytest.mark.ragas
@pytest.mark.integration
def test_automerging_evaluation_runs(ragas_result_automerging: Any) -> None:
    """
    Verifica que la evaluación AutoMerging se ejecuta y retorna las
    cuatro métricas esperadas sin errores.
    """
    df = ragas_result_automerging.to_pandas()
    assert not df.empty, "El resultado de evaluación AutoMerging no debe estar vacío."
    for metrica in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        assert metrica in df.columns, (
            f"La métrica '{metrica}' no está presente en el resultado AutoMerging."
        )


@pytest.mark.ragas
@pytest.mark.integration
def test_automerging_faithfulness(ragas_result_automerging: Any) -> None:
    """
    Gate AutoMerging — Faithfulness >= MIN_FAITHFULNESS.

    Valida que el contexto mergeado no induzca alucinaciones en la respuesta.
    Una baja faithfulness en esta sección sugiere que el merge incluye nodos
    no relacionados o que el nodo padre aporta ruido.
    """
    score = ragas_result_automerging.to_pandas()["faithfulness"].mean()
    assert score >= MIN_FAITHFULNESS, (
        f"[AutoMerging] Faithfulness ({score:.3f}) por debajo del umbral ({MIN_FAITHFULNESS}). "
        "Revisar simple_ratio_thresh del AutoMergingRetriever o la fragmentación jerárquica."
    )


@pytest.mark.ragas
@pytest.mark.integration
def test_automerging_answer_relevancy(ragas_result_automerging: Any) -> None:
    """
    Gate AutoMerging — AnswerRelevancy >= MIN_ANSWER_RELEVANCY.

    Valida que las respuestas generadas con contexto mergeado sean pertinentes
    a las preguntas que abarcan tanto descripción como comentarios del issue.
    """
    score = ragas_result_automerging.to_pandas()["answer_relevancy"].mean()
    assert score >= MIN_ANSWER_RELEVANCY, (
        f"[AutoMerging] AnswerRelevancy ({score:.3f}) por debajo del umbral ({MIN_ANSWER_RELEVANCY}). "
        "El retriever DENSE podría no recuperar los nodos hoja correctos para triggerear el merge."
    )


@pytest.mark.ragas
@pytest.mark.integration
def test_automerging_context_precision(ragas_result_automerging: Any) -> None:
    """
    Gate AutoMerging — ContextPrecision >= MIN_CONTEXT_PRECISION.

    Valida que los nodos padre mergeados aparezcan en las primeras posiciones
    del ranking, no desplazados por nodos hoja irrelevantes.
    """
    score = ragas_result_automerging.to_pandas()["context_precision"].mean()
    assert score >= MIN_CONTEXT_PRECISION, (
        f"[AutoMerging] ContextPrecision ({score:.3f}) por debajo del umbral ({MIN_CONTEXT_PRECISION}). "
        "Considerar aumentar similarity_top_k o ajustar simple_ratio_thresh."
    )


@pytest.mark.ragas
@pytest.mark.integration
def test_automerging_context_recall(ragas_result_automerging: Any) -> None:
    """
    Gate AutoMerging — ContextRecall >= MIN_CONTEXT_RECALL.

    Valida que el AutoMergingRetriever recupere el contexto completo necesario
    (descripción + comentarios) para responder preguntas que abarcan el issue
    de punta a punta.
    """
    score = ragas_result_automerging.to_pandas()["context_recall"].mean()
    assert score >= MIN_CONTEXT_RECALL, (
        f"[AutoMerging] ContextRecall ({score:.3f}) por debajo del umbral ({MIN_CONTEXT_RECALL}). "
        "Revisar la fragmentación HierarchicalNodeParser o el ratio de merge."
    )


# ---------------------------------------------------------------------------
# Sección B — Hybrid (AutoMerging + BM25)
# Retriever: VectorIndexAutoRetriever (HYBRID, alpha=0.5) + AutoMergingRetriever
# Casos: preguntas con nombres de método, flags y siglas exactas que requieren
#        la rama léxica BM25 para recuperar los nodos correctos.
# ---------------------------------------------------------------------------


@pytest.mark.ragas
@pytest.mark.integration
def test_hybrid_evaluation_runs(ragas_result_hybrid: Any) -> None:
    """
    Verifica que la evaluación Hybrid se ejecuta y retorna las
    cuatro métricas esperadas sin errores.
    """
    df = ragas_result_hybrid.to_pandas()
    assert not df.empty, "El resultado de evaluación Hybrid no debe estar vacío."
    for metrica in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        assert metrica in df.columns, (
            f"La métrica '{metrica}' no está presente en el resultado Hybrid."
        )


@pytest.mark.ragas
@pytest.mark.integration
def test_hybrid_faithfulness(ragas_result_hybrid: Any) -> None:
    """
    Gate Hybrid — Faithfulness >= MIN_FAITHFULNESS.

    Valida que la combinación BM25 + embeddings no recupere nodos falsos positivos
    que induzcan alucinaciones en la respuesta. Un score bajo puede indicar que
    el peso BM25 (alpha < 0.5) recupera demasiados fragmentos irrelevantes.
    """
    score = ragas_result_hybrid.to_pandas()["faithfulness"].mean()
    assert score >= MIN_FAITHFULNESS, (
        f"[Hybrid] Faithfulness ({score:.3f}) por debajo del umbral ({MIN_FAITHFULNESS}). "
        "Revisar el parámetro alpha (balance denso↔sparse) o sparse_top_k en el retriever."
    )


@pytest.mark.ragas
@pytest.mark.integration
def test_hybrid_answer_relevancy(ragas_result_hybrid: Any) -> None:
    """
    Gate Hybrid — AnswerRelevancy >= MIN_ANSWER_RELEVANCY.

    Valida que la búsqueda léxica BM25 mejore (o al menos no degrade) la
    pertinencia de las respuestas cuando las preguntas contienen términos
    técnicos exactos como nombres de método o flags de CLI.
    """
    score = ragas_result_hybrid.to_pandas()["answer_relevancy"].mean()
    assert score >= MIN_ANSWER_RELEVANCY, (
        f"[Hybrid] AnswerRelevancy ({score:.3f}) por debajo del umbral ({MIN_ANSWER_RELEVANCY}). "
        "Verificar que el modelo Qdrant/bm25 esté correctamente configurado en el vector store."
    )


@pytest.mark.ragas
@pytest.mark.integration
def test_hybrid_context_precision(ragas_result_hybrid: Any) -> None:
    """
    Gate Hybrid — ContextPrecision >= MIN_CONTEXT_PRECISION.

    Valida que los fragmentos con los términos léxicos exactos aparezcan
    primero en el ranking. La rama BM25 debería elevar los nodos con match
    exacto de siglas o nombres de función.
    """
    score = ragas_result_hybrid.to_pandas()["context_precision"].mean()
    assert score >= MIN_CONTEXT_PRECISION, (
        f"[Hybrid] ContextPrecision ({score:.3f}) por debajo del umbral ({MIN_CONTEXT_PRECISION}). "
        "Revisar la configuración HNSW de Qdrant o el parámetro alpha del retriever híbrido."
    )


@pytest.mark.ragas
@pytest.mark.integration
def test_hybrid_context_recall(ragas_result_hybrid: Any) -> None:
    """
    Gate Hybrid — ContextRecall >= MIN_CONTEXT_RECALL.

    Valida que el retriever híbrido recupere todos los fragmentos necesarios
    para responder preguntas con terminología técnica muy específica. Una baja
    recall puede indicar que BM25 filtra nodos relevantes por falta de match
    léxico exacto (ej. stemming o tokenización).
    """
    score = ragas_result_hybrid.to_pandas()["context_recall"].mean()
    assert score >= MIN_CONTEXT_RECALL, (
        f"[Hybrid] ContextRecall ({score:.3f}) por debajo del umbral ({MIN_CONTEXT_RECALL}). "
        "Aumentar sparse_top_k o ajustar alpha para dar más peso a los embeddings densos."
    )


# ---------------------------------------------------------------------------
# Sección C — AutoMerging + Hybrid + Metadata Filtering
# Retriever: VectorIndexAutoRetriever (HYBRID) + AutoMergingRetriever
# Filtros: inferidos implícitamente del lenguaje natural de cada pregunta.
# Casos: preguntas que mencionan project, status o author explícitamente,
#        lo que el VectorIndexAutoRetriever debe traducir a filtros de metadata.
# ---------------------------------------------------------------------------


@pytest.mark.ragas
@pytest.mark.integration
def test_metadata_filter_evaluation_runs(ragas_result_metadata_filter: Any) -> None:
    """
    Verifica que la evaluación con Metadata Filtering se ejecuta y retorna
    las cuatro métricas esperadas sin errores.
    """
    df = ragas_result_metadata_filter.to_pandas()
    assert not df.empty, "El resultado de evaluación Metadata Filtering no debe estar vacío."
    for metrica in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        assert metrica in df.columns, (
            f"La métrica '{metrica}' no está presente en el resultado Metadata Filtering."
        )


@pytest.mark.ragas
@pytest.mark.integration
def test_metadata_filter_faithfulness(ragas_result_metadata_filter: Any) -> None:
    """
    Gate Metadata Filtering — Faithfulness >= MIN_FAITHFULNESS.

    Valida que los filtros de metadatos inferidos no eliminen nodos relevantes
    y que la respuesta generada se base únicamente en el contexto recuperado.
    Un score bajo puede indicar que el LLM del auto-retriever genera filtros
    incorrectos que devuelven fragmentos de proyectos equivocados.
    """
    score = ragas_result_metadata_filter.to_pandas()["faithfulness"].mean()
    assert score >= MIN_FAITHFULNESS, (
        f"[MetadataFilter] Faithfulness ({score:.3f}) por debajo del umbral ({MIN_FAITHFULNESS}). "
        "Revisar REDMINE_VECTOR_STORE_INFO o el prompt de inferencia del VectorIndexAutoRetriever."
    )


@pytest.mark.ragas
@pytest.mark.integration
def test_metadata_filter_answer_relevancy(ragas_result_metadata_filter: Any) -> None:
    """
    Gate Metadata Filtering — AnswerRelevancy >= MIN_ANSWER_RELEVANCY.

    Valida que las respuestas sean pertinentes cuando el retriever filtra por
    project/status/author. Un score bajo puede indicar que el filtro elimina
    todos los nodos relevantes (over-filtering) y el LLM responde sin contexto.
    """
    score = ragas_result_metadata_filter.to_pandas()["answer_relevancy"].mean()
    assert score >= MIN_ANSWER_RELEVANCY, (
        f"[MetadataFilter] AnswerRelevancy ({score:.3f}) por debajo del umbral ({MIN_ANSWER_RELEVANCY}). "
        "Verificar que las descripciones de MetadataInfo en schemas.py sean lo suficientemente claras."
    )


@pytest.mark.ragas
@pytest.mark.integration
def test_metadata_filter_context_precision(ragas_result_metadata_filter: Any) -> None:
    """
    Gate Metadata Filtering — ContextPrecision >= MIN_CONTEXT_PRECISION.

    Valida que, al aplicar filtros de metadatos, los fragmentos recuperados
    sean altamente relevantes (menos ruido de otros proyectos/autores).
    El filtrado correcto debería mejorar la precision respecto a la búsqueda sin filtros.
    """
    score = ragas_result_metadata_filter.to_pandas()["context_precision"].mean()
    assert score >= MIN_CONTEXT_PRECISION, (
        f"[MetadataFilter] ContextPrecision ({score:.3f}) por debajo del umbral ({MIN_CONTEXT_PRECISION}). "
        "El VectorIndexAutoRetriever podría no estar infiriendo filtros de metadata correctamente."
    )


@pytest.mark.ragas
@pytest.mark.integration
def test_metadata_filter_context_recall(ragas_result_metadata_filter: Any) -> None:
    """
    Gate Metadata Filtering — ContextRecall >= MIN_CONTEXT_RECALL.

    Valida que el filtrado de metadatos no cause pérdida de recall (que no se
    excluyan fragmentos relevantes por filtros demasiado restrictivos).
    Un score bajo puede indicar que el auto-retriever genera filtros que
    excluyen nodos padre necesarios para el merge jerárquico.
    """
    score = ragas_result_metadata_filter.to_pandas()["context_recall"].mean()
    assert score >= MIN_CONTEXT_RECALL, (
        f"[MetadataFilter] ContextRecall ({score:.3f}) por debajo del umbral ({MIN_CONTEXT_RECALL}). "
        "Los filtros inferidos podrían estar excluyendo nodos de comentarios (entity_type=issue_comment) "
        "que no tienen el campo 'status' o 'author' en sus metadatos."
    )
