"""Paquete centralizado para interacción con LLMs y mecanismos de fallback."""

from src.llm.fallback_llm import (
    FallbackLLM,
    FallbackStructuredOutput,
    construir_cadena_proveedores,
)

__all__ = [
    "FallbackLLM",
    "FallbackStructuredOutput",
    "construir_cadena_proveedores",
]
