"""
src/rag/utils/get_embedding.py
==============================
Registro y fábrica de modelos de embeddings para LlamaIndex.
Soporta ejecución local (HuggingFace/sentence-transformers), API remota
(HuggingFace Inference API), Ollama y OpenAI.
"""

from __future__ import annotations

import logging
import os
from typing import Callable

from dotenv import load_dotenv
from llama_index.core.base.embeddings.base import BaseEmbedding

load_dotenv()

logger = logging.getLogger(__name__)


def _get_hf_local_embedding(model_name: str | None = None) -> BaseEmbedding:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    resolved_model = model_name or os.getenv("EMBED_MODEL", "BAAI/bge-m3")
    logger.info("Inicializando embeddings locales HuggingFace (modelo: %s)", resolved_model)
    return HuggingFaceEmbedding(model_name=resolved_model)


def _get_hf_api_embedding(model_name: str | None = None) -> BaseEmbedding:
    from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding

    resolved_model = model_name or os.getenv("EMBED_MODEL", "BAAI/bge-m3")
    token = os.getenv("HUGGINGFACE_API_KEY", "")
    logger.info("Inicializando embeddings HuggingFace Inference API (modelo: %s)", resolved_model)
    return HuggingFaceInferenceAPIEmbedding(
        model_name=resolved_model,
        token=token,
    )


def _get_ollama_embedding(model_name: str | None = None) -> BaseEmbedding:
    from llama_index.embeddings.ollama import OllamaEmbedding

    resolved_model = model_name or os.getenv("EMBED_MODEL", "bge-m3")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    logger.info("Inicializando embeddings Ollama (modelo: %s, url: %s)", resolved_model, base_url)
    return OllamaEmbedding(model_name=resolved_model, base_url=base_url)


def _get_openai_embedding(model_name: str | None = None) -> BaseEmbedding:
    from llama_index.embeddings.openai import OpenAIEmbedding

    resolved_model = model_name or os.getenv("EMBED_MODEL", "text-embedding-3-small")
    api_key = os.getenv("OPENAI_API_KEY", "")
    logger.info("Inicializando embeddings OpenAI (modelo: %s)", resolved_model)
    return OpenAIEmbedding(model_name=resolved_model, api_key=api_key)


_EMBEDDING_REGISTRY: dict[str, Callable[[str | None], BaseEmbedding]] = {
    "local": _get_hf_local_embedding,
    "huggingface": _get_hf_local_embedding,
    "sentence_transformers": _get_hf_local_embedding,
    "hf_api": _get_hf_api_embedding,
    "huggingface_api": _get_hf_api_embedding,
    "huggingface-api": _get_hf_api_embedding,
    "ollama": _get_ollama_embedding,
    "openai": _get_openai_embedding,
}


def get_embedding_model(
    provider: str | None = None,
    model_name: str | None = None,
) -> BaseEmbedding:
    """
    Retorna la instancia del modelo de embedding según el proveedor especificado.
    Prioridad de proveedor: argumento `provider` > variable `EMBEDDING_PROVIDER` > default 'local'.
    """
    resolved_provider = (provider or os.getenv("EMBEDDING_PROVIDER", "local")).lower().strip()
    factory = _EMBEDDING_REGISTRY.get(resolved_provider)
    if factory is None:
        valid_providers = ", ".join(sorted(set(_EMBEDDING_REGISTRY.keys())))
        raise ValueError(
            f"Proveedor de embedding no soportado: '{resolved_provider}'. "
            f"Opciones válidas: {valid_providers}"
        )
    return factory(model_name)
