"""Pruebas unitarias para FallbackLLM y utilidades en src/llm/fallback_llm.py."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from llama_index.core.llms import CompletionResponse

from src.llm.fallback_llm import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_NVIDIA_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    FallbackLLM,
    FallbackStructuredOutput,
    _es_error_429,
    _extraer_uso_tokens,
)


class DummyHTTPError(Exception):
    def __init__(self, status_code: int, message: str = "") -> None:
        super().__init__(message)
        self.response = MagicMock(status_code=status_code)


class SampleSchema(BaseModel):
    summary: str


def test_es_error_429():
    assert _es_error_429(Exception("Error: 429 Too Many Requests"))
    assert _es_error_429(Exception("Rate limit exceeded"))
    assert _es_error_429(Exception("ratelimit reached"))
    assert _es_error_429(DummyHTTPError(429, "rate limited"))
    assert not _es_error_429(DummyHTTPError(500, "internal error"))
    assert not _es_error_429(Exception("Generic connection error"))


def test_extraer_uso_tokens_llamaindex():
    resp = CompletionResponse(
        text="ok",
        raw={"usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}},
    )
    prompt, comp, total = _extraer_uso_tokens(resp)
    assert prompt == 20
    assert comp == 10
    assert total == 30


def test_extraer_uso_tokens_langchain():
    class DummyLangChainMessage:
        def __init__(self):
            self.usage_metadata = {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}

    prompt, comp, total = _extraer_uso_tokens(DummyLangChainMessage())
    assert prompt == 12
    assert comp == 8
    assert total == 20


def test_fallback_llm_defaults():
    llm = FallbackLLM(proveedores=[])
    assert llm.timeout_segundos == 60
    assert llm.proveedores == []
    assert llm.metadata.model_name == "fallback-llm"


def test_fallback_llm_cascada_sync():
    # Proveedor 1 falla con 429
    p1 = MagicMock()
    p1.complete.side_effect = Exception("429 rate limit")
    p1.model = DEFAULT_GROQ_MODEL

    # Proveedor 2 tiene éxito
    p2 = MagicMock()
    p2.complete.return_value = CompletionResponse(
        text="respuesta exitosa",
        raw={"usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}},
    )
    p2.model = DEFAULT_NVIDIA_MODEL

    llm = FallbackLLM(proveedores=[p1, p2], timeout_segundos=5)
    resp = llm.complete("hola")

    assert resp.text == "respuesta exitosa"
    assert p1.complete.call_count == 1
    assert p2.complete.call_count == 1
    assert "llamadas=1" in llm.resumen_consumo()


@pytest.mark.asyncio
async def test_fallback_llm_cascada_async():
    # Proveedor 1 falla con 429
    p1 = MagicMock()

    async def fail_acomplete(*args, **kwargs):
        raise Exception("Too Many Requests 429")

    p1.acomplete = fail_acomplete
    p1.model = DEFAULT_GROQ_MODEL

    # Proveedor 2 tiene éxito
    p2 = MagicMock()

    async def ok_acomplete(*args, **kwargs):
        return CompletionResponse(text="respuesta async exitosa")

    p2.acomplete = ok_acomplete
    p2.model = DEFAULT_OPENROUTER_MODEL

    llm = FallbackLLM(proveedores=[p1, p2], timeout_segundos=5)
    resp = await llm.acomplete("hola async")

    assert resp.text == "respuesta async exitosa"


def test_kwargs_sanitizer():
    def custom_sanitizer(kwargs: dict[str, Any]) -> dict[str, Any]:
        kw = dict(kwargs)
        kw["n"] = 1
        return kw

    p = MagicMock()
    p.complete.return_value = CompletionResponse(text="ok")

    llm = FallbackLLM(proveedores=[p], kwargs_sanitizer=custom_sanitizer)
    llm.complete("test", n=5)

    assert p.complete.call_args.kwargs["n"] == 1


def test_bind_tools():
    p1 = MagicMock()
    p1.bind_tools = MagicMock(return_value="bound_p1")
    p2 = MagicMock(spec=[])  # no tiene bind_tools

    llm = FallbackLLM(proveedores=[p1, p2])
    llm_with_tools = llm.bind_tools(["tool_a", "tool_b"])

    assert isinstance(llm_with_tools, FallbackLLM)
    p1.bind_tools.assert_called_once_with(["tool_a", "tool_b"])
    assert llm_with_tools.proveedores == ["bound_p1", p2]


def test_with_structured_output():
    class MockStructuredRunner:
        def invoke(self, input_data, **kwargs):
            return SampleSchema(summary="salida estructurada ok")

    p1 = MagicMock()
    p1.with_structured_output.side_effect = Exception("429 rate limit")

    p2 = MagicMock()
    p2.with_structured_output.return_value = MockStructuredRunner()

    llm = FallbackLLM(proveedores=[p1, p2])
    structured_llm = llm.with_structured_output(SampleSchema)

    assert isinstance(structured_llm, FallbackStructuredOutput)
    res = structured_llm.invoke("analizar")
    assert res.summary == "salida estructurada ok"
