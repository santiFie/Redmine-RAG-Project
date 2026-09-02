"""Módulo centralizado de LLM con fallback en cascada para tolerancia a fallos.

Proporciona `FallbackLLM`, una implementación resiliente que ejecuta llamadas en
cascada (Groq → NVIDIA → OpenRouter u otros proveedores personalizados) gestionando
automáticamente errores de rate limit (HTTP 429) y timeouts.

Es compatible tanto con el ecosistema de LlamaIndex (RAG y Ragas) como con los agentes
de LangGraph mediante `bind_tools`, `with_structured_output`, `invoke` y `ainvoke`.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import os
from collections.abc import Callable, Sequence
from typing import Any

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

logger = logging.getLogger(__name__)

# Modelos y parámetros por defecto
DEFAULT_GROQ_MODEL: str = "openai/gpt-oss-120b"
DEFAULT_NVIDIA_MODEL: str = "openai/gpt-oss-120b"
DEFAULT_OPENROUTER_MODEL: str = "openai/gpt-5.4-nano"
DEFAULT_TIMEOUT_SEGUNDOS: int = 60


def _es_error_429(exc: BaseException) -> bool:
    """Determina si la excepción corresponde a un error de rate limit (HTTP 429).

    Detecta:
      - Excepciones cuyo mensaje contiene '429', 'rate limit' o 'too many requests'.
      - httpx.HTTPStatusError y similares que expongan ``.response.status_code``.
    """
    msg = str(exc).lower()
    if any(kw in msg for kw in ("rate limit", "429", "too many requests", "ratelimit")):
        return True
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 429:
        return True
    return False


def _extraer_uso_tokens(respuesta: Any) -> tuple[int, int, int]:
    """Extrae (prompt_tokens, completion_tokens, total_tokens) de una respuesta LLM.

    Soporta:
      - Proveedores de LlamaIndex compatibles con la API de OpenAI (``respuesta.raw['usage']``
        o ``respuesta.additional_kwargs['usage']``).
      - Mensajes de LangChain (``usage_metadata`` o ``response_metadata['token_usage']``).
    """
    uso: dict[str, Any] = {}

    # LangChain usage_metadata (AIMessage.usage_metadata)
    usage_metadata = getattr(respuesta, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        p = int(usage_metadata.get("input_tokens") or 0)
        c = int(usage_metadata.get("output_tokens") or 0)
        t = int(usage_metadata.get("total_tokens") or (p + c))
        return (p, c, t)

    # LangChain response_metadata
    resp_meta = getattr(respuesta, "response_metadata", None)
    if isinstance(resp_meta, dict):
        uso = resp_meta.get("token_usage") or resp_meta.get("usage") or {}

    # LlamaIndex raw['usage']
    if not uso:
        raw = getattr(respuesta, "raw", None)
        if isinstance(raw, dict):
            uso = raw.get("usage") or {}

    # LlamaIndex additional_kwargs['usage']
    if not uso:
        kwargs_adicionales = getattr(respuesta, "additional_kwargs", None)
        if isinstance(kwargs_adicionales, dict):
            uso = kwargs_adicionales.get("usage") or {}

    return (
        int(uso.get("prompt_tokens") or uso.get("input_tokens") or 0),
        int(uso.get("completion_tokens") or uso.get("output_tokens") or 0),
        int(uso.get("total_tokens") or 0),
    )


def construir_cadena_proveedores(
    groq_model: str = DEFAULT_GROQ_MODEL,
    nvidia_model: str = DEFAULT_NVIDIA_MODEL,
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL,
    max_retries: int = 1,
) -> list[Any]:
    """Construye la cadena de proveedores LLM en orden de prioridad.

    Orden:
      1. Groq       → groq_model
      2. NVIDIA     → nvidia_model
      3. OpenRouter → openrouter_model

    Los proveedores cuyos paquetes no estén instalados se omiten con advertencia.
    """
    proveedores: list[Any] = []

    try:
        from llama_index.llms.groq import Groq  # noqa: PLC0415

        proveedores.append(
            Groq(
                model=groq_model,
                api_key=os.getenv("GROQ_API_KEY", ""),
                max_retries=max_retries,
            )
        )
    except ImportError:
        logger.warning("llama-index-llms-groq no instalado — proveedor Groq omitido.")

    try:
        from llama_index.llms.nvidia import NVIDIA  # noqa: PLC0415

        proveedores.append(
            NVIDIA(
                model=nvidia_model,
                api_key=os.getenv("NVIDIA_API_KEY", ""),
                max_retries=max_retries,
            )
        )
    except ImportError:
        logger.warning("llama-index-llms-nvidia no instalado — fallback NVIDIA omitido.")

    try:
        from llama_index.llms.openai_like import OpenAILike  # noqa: PLC0415

        proveedores.append(
            OpenAILike(
                model=openrouter_model,
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                api_base="https://openrouter.ai/api/v1",
                is_chat_model=True,
                max_retries=max_retries,
            )
        )
    except ImportError:
        logger.warning("llama-index-llms-openai-like no instalado — fallback OpenRouter omitido.")

    cadena = " → ".join(
        f"{type(p).__name__}(modelo={getattr(p, 'model', getattr(p, 'model_name', '?'))})"
        for p in proveedores
    )
    logger.info("Cadena de fallback LLM construida: %s", cadena)
    return proveedores


class FallbackStructuredOutput:
    """Wrapper de salida estructurada con fallback en cascada ante errores 429 y timeout."""

    def __init__(
        self,
        fallback_llm: FallbackLLM,
        schema: Any,
        **kwargs: Any,
    ) -> None:
        self._fallback_llm = fallback_llm
        self._schema = schema
        self._kwargs = kwargs

    def invoke(self, input_data: Any, **kwargs: Any) -> Any:
        """Ejecuta la extracción estructurada síncrona en cascada."""
        ultimo_error: BaseException = RuntimeError("No hay proveedores disponibles.")
        timeout = self._fallback_llm.timeout_segundos

        for proveedor in self._fallback_llm.proveedores:
            try:
                # Obtener el runnable con structured output del proveedor
                if hasattr(proveedor, "with_structured_output"):
                    runner = proveedor.with_structured_output(self._schema, **self._kwargs)
                elif hasattr(proveedor, "as_structured_llm"):
                    runner = proveedor.as_structured_llm(self._schema, **self._kwargs)
                else:
                    logger.warning(
                        "Proveedor %s no soporta with_structured_output — saltando.",
                        type(proveedor).__name__,
                    )
                    continue

                fn = getattr(runner, "invoke", None) or getattr(runner, "chat", None)
                if fn is None:
                    continue

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    futuro = pool.submit(fn, input_data, **kwargs)
                    respuesta = futuro.result(timeout=timeout)

                self._fallback_llm.registrar_exito(proveedor, "structured_invoke", respuesta)
                return respuesta
            except (concurrent.futures.TimeoutError, TimeoutError) as exc:
                ultimo_error = exc
                logger.warning(
                    "Timeout (%ds) en structured_invoke con %s — pasando al siguiente proveedor.",
                    timeout,
                    type(proveedor).__name__,
                )
            except Exception as exc:
                if _es_error_429(exc):
                    ultimo_error = exc
                    logger.warning(
                        "Rate limit (429) en structured_invoke con %s — pasando al siguiente.",
                        type(proveedor).__name__,
                    )
                else:
                    raise

        logger.error(
            "Todos los proveedores fallaron en 'structured_invoke'. Último error: %s",
            ultimo_error,
        )
        raise RuntimeError(
            "Todos los proveedores fallaron en 'structured_invoke'."
        ) from ultimo_error

    async def ainvoke(self, input_data: Any, **kwargs: Any) -> Any:
        """Ejecuta la extracción estructurada asíncrona en cascada."""
        ultimo_error: BaseException = RuntimeError("No hay proveedores disponibles.")
        timeout = self._fallback_llm.timeout_segundos

        for proveedor in self._fallback_llm.proveedores:
            try:
                if hasattr(proveedor, "with_structured_output"):
                    runner = proveedor.with_structured_output(self._schema, **self._kwargs)
                elif hasattr(proveedor, "as_structured_llm"):
                    runner = proveedor.as_structured_llm(self._schema, **self._kwargs)
                else:
                    logger.warning(
                        "Proveedor %s no soporta with_structured_output — saltando.",
                        type(proveedor).__name__,
                    )
                    continue

                fn = getattr(runner, "ainvoke", None) or getattr(runner, "achat", None)
                if fn is None:
                    continue

                coro = fn(input_data, **kwargs)
                respuesta = await asyncio.wait_for(coro, timeout=float(timeout))
                self._fallback_llm.registrar_exito(proveedor, "structured_ainvoke", respuesta)
                return respuesta
            except TimeoutError as exc:
                ultimo_error = exc
                logger.warning(
                    "Timeout async (%ds) en structured_ainvoke con %s — pasando al siguiente.",
                    timeout,
                    type(proveedor).__name__,
                )
            except Exception as exc:
                if _es_error_429(exc):
                    ultimo_error = exc
                    logger.warning(
                        "Rate limit (429) async en structured_ainvoke con %s — pasando al siguiente.",
                        type(proveedor).__name__,
                    )
                else:
                    raise

        logger.error(
            "Todos los proveedores fallaron (async) en 'structured_ainvoke'. Último error: %s",
            ultimo_error,
        )
        raise RuntimeError(
            "Todos los proveedores async fallaron en 'structured_ainvoke'."
        ) from ultimo_error


class FallbackLLM(LLM):
    """LLM evaluador y orquestador con fallback en cascada para errores 429 (rate limit) y timeout.

    Compatible tanto con la interfaz de LlamaIndex (``complete``, ``chat``, etc.) como con
    LangGraph / LangChain (``bind_tools``, ``with_structured_output``, ``invoke``, ``ainvoke``).

    Por cada llamada se intenta cada proveedor en orden. Si se recibe un error 429
    o un timeout se pasa inmediatamente al siguiente sin backoff.
    Si todos los proveedores fallan, se propaga el último error.
    """

    _proveedores: list[Any] = PrivateAttr(default_factory=list)
    _timeout_segundos: int = PrivateAttr(default=DEFAULT_TIMEOUT_SEGUNDOS)
    _kwargs_sanitizer: Callable[[dict[str, Any]], dict[str, Any]] | None = PrivateAttr(default=None)
    _llamadas_totales: int = PrivateAttr(default=0)
    _tokens_prompt: int = PrivateAttr(default=0)
    _tokens_completion: int = PrivateAttr(default=0)
    _tokens_total: int = PrivateAttr(default=0)

    def __init__(
        self,
        proveedores: Sequence[Any] | None = None,
        groq_model: str = DEFAULT_GROQ_MODEL,
        nvidia_model: str = DEFAULT_NVIDIA_MODEL,
        openrouter_model: str = DEFAULT_OPENROUTER_MODEL,
        timeout_segundos: int = DEFAULT_TIMEOUT_SEGUNDOS,
        kwargs_sanitizer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Inicializa FallbackLLM.

        Args:
            proveedores: Lista opcional de instancias de proveedores LLM. Si no se indica,
                se construye la cadena por defecto con los modelos configurados.
            groq_model: Nombre del modelo para Groq si se construye la cadena por defecto.
            nvidia_model: Nombre del modelo para NVIDIA si se construye la cadena por defecto.
            openrouter_model: Nombre del modelo para OpenRouter si se construye la cadena por defecto.
            timeout_segundos: Tiempo límite en segundos por intento por cada proveedor.
            kwargs_sanitizer: Función opcional para transformar/sanitizar kwargs antes
                de pasarlos a los proveedores subyacentes (e.g. forzar n=1 en Ragas).
            **kwargs: Argumentos adicionales pasados a la clase base de LlamaIndex.
        """
        super().__init__(**kwargs)
        if proveedores is not None:
            self._proveedores = list(proveedores)
        else:
            self._proveedores = construir_cadena_proveedores(
                groq_model=groq_model,
                nvidia_model=nvidia_model,
                openrouter_model=openrouter_model,
            )
        self._timeout_segundos = timeout_segundos
        self._kwargs_sanitizer = kwargs_sanitizer

    @property
    def proveedores(self) -> list[Any]:
        """Lista de proveedores configurados en la cadena de fallback."""
        return self._proveedores

    @property
    def timeout_segundos(self) -> int:
        """Tiempo límite en segundos por intento por proveedor."""
        return self._timeout_segundos

    # ------------------------------------------------------------------
    # Observabilidad: registro de éxito y consumo acumulado
    # ------------------------------------------------------------------

    def registrar_exito(self, proveedor: Any, nombre_metodo: str, respuesta: Any) -> None:
        """Registra una llamada LLM exitosa: acumula tokens consumidos y emite un log."""
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
            getattr(proveedor, "model", getattr(proveedor, "model_name", "?")),
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
    # Integración con LangGraph y herramientas (bind_tools / with_structured_output)
    # ------------------------------------------------------------------

    def bind_tools(self, tools: list, **kwargs: Any) -> FallbackLLM:
        """Vincula herramientas a cada proveedor de la cadena que lo soporte."""
        nuevos_proveedores = []
        for prov in self._proveedores:
            if hasattr(prov, "bind_tools"):
                nuevos_proveedores.append(prov.bind_tools(tools, **kwargs))
            else:
                nuevos_proveedores.append(prov)

        return FallbackLLM(
            proveedores=nuevos_proveedores,
            timeout_segundos=self._timeout_segundos,
            kwargs_sanitizer=self._kwargs_sanitizer,
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> FallbackStructuredOutput:
        """Devuelve un wrapper de salida estructurada con fallback en cascada."""
        return FallbackStructuredOutput(self, schema, **kwargs)

    # ------------------------------------------------------------------
    # Helpers internos de fallback
    # ------------------------------------------------------------------

    def _preparar_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Aplica el sanitizador/transformador de kwargs si fue configurado."""
        if self._kwargs_sanitizer is not None:
            return self._kwargs_sanitizer(kwargs)
        return dict(kwargs)

    def _intentar_sync(self, nombre_metodo: str, *args: Any, **kwargs: Any) -> Any:
        """Ejecuta ``nombre_metodo`` en cada proveedor con timeout. Sin backoff entre intentos."""
        kwargs_limpios = self._preparar_kwargs(kwargs)
        ultimo_error: BaseException = RuntimeError("No hay proveedores disponibles.")

        for proveedor in self._proveedores:
            try:
                fn = getattr(proveedor, nombre_metodo, None)
                if fn is None:
                    continue
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    futuro = pool.submit(fn, *args, **kwargs_limpios)
                    respuesta = futuro.result(timeout=self._timeout_segundos)
                self.registrar_exito(proveedor, nombre_metodo, respuesta)
                return respuesta
            except (concurrent.futures.TimeoutError, TimeoutError) as exc:
                ultimo_error = exc
                logger.warning(
                    "Timeout (%ds) en %s(modelo=%s).%s — pasando al siguiente proveedor.",
                    self._timeout_segundos,
                    type(proveedor).__name__,
                    getattr(proveedor, "model", getattr(proveedor, "model_name", "?")),
                    nombre_metodo,
                )
            except Exception as exc:
                if _es_error_429(exc):
                    ultimo_error = exc
                    logger.warning(
                        "Rate limit (429) en %s(modelo=%s).%s — pasando al siguiente proveedor.",
                        type(proveedor).__name__,
                        getattr(proveedor, "model", getattr(proveedor, "model_name", "?")),
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
        """Variante async de ``_intentar_sync`` con ``asyncio.wait_for``."""
        kwargs_limpios = self._preparar_kwargs(kwargs)
        ultimo_error: BaseException = RuntimeError("No hay proveedores disponibles.")

        for proveedor in self._proveedores:
            try:
                fn = getattr(proveedor, nombre_metodo, None)
                if fn is None:
                    continue
                coro = fn(*args, **kwargs_limpios)
                respuesta = await asyncio.wait_for(coro, timeout=float(self._timeout_segundos))
                self.registrar_exito(proveedor, nombre_metodo, respuesta)
                return respuesta
            except TimeoutError as exc:
                ultimo_error = exc
                logger.warning(
                    "Timeout async (%ds) en %s(modelo=%s).%s — pasando al siguiente proveedor.",
                    self._timeout_segundos,
                    type(proveedor).__name__,
                    getattr(proveedor, "model", getattr(proveedor, "model_name", "?")),
                    nombre_metodo,
                )
            except Exception as exc:
                if _es_error_429(exc):
                    ultimo_error = exc
                    logger.warning(
                        "Rate limit (429) async en %s(modelo=%s).%s — pasando al siguiente.",
                        type(proveedor).__name__,
                        getattr(proveedor, "model", getattr(proveedor, "model_name", "?")),
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
    # Métodos LangGraph / LangChain directos (invoke / ainvoke)
    # ------------------------------------------------------------------

    def invoke(self, input_data: Any, **kwargs: Any) -> Any:
        """Ejecuta una invocación síncrona en cascada compatible con LangChain/LangGraph."""
        return self._intentar_sync("invoke", input_data, **kwargs)

    async def ainvoke(self, input_data: Any, **kwargs: Any) -> Any:
        """Ejecuta una invocación asíncrona en cascada compatible con LangChain/LangGraph."""
        return await self._intentar_async("ainvoke", input_data, **kwargs)

    # ------------------------------------------------------------------
    # Métodos abstractos requeridos por LlamaIndex LLM (sync)
    # ------------------------------------------------------------------

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        """Completa el prompt con fallback en cascada."""
        return self._intentar_sync("complete", prompt, formatted=formatted, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Delega a ``complete`` sin streaming real."""
        resp = self.complete(prompt, formatted=formatted, **kwargs)
        yield resp

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat con fallback en cascada."""
        return self._intentar_sync("chat", messages, **kwargs)

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        """Delega a ``chat`` sin streaming real."""
        resp = self.chat(messages, **kwargs)
        yield resp

    # ------------------------------------------------------------------
    # Métodos async requeridos por LlamaIndex LLM
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
        """Delega a ``achat`` sin streaming real."""

        async def _gen():
            yield await self.achat(messages, **kwargs)

        return _gen()

    @llm_completion_callback()
    def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        """Delega a ``acomplete`` sin streaming real."""

        async def _gen():
            yield await self.acomplete(prompt, formatted=formatted, **kwargs)

        return _gen()
