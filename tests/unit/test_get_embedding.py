"""
tests/unit/test_get_embedding.py
================================
Pruebas unitarias para la fábrica de embeddings (get_embedding_model).
"""

import pytest
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from src.rag.utils.get_embedding import get_embedding_model


def test_get_embedding_model_local():
    """Verifica que el proveedor 'local' / 'huggingface' instancie HuggingFaceEmbedding."""
    model = get_embedding_model("local", "BAAI/bge-m3")
    assert isinstance(model, HuggingFaceEmbedding)
    assert model.model_name == "BAAI/bge-m3"


def test_get_embedding_model_hf_api(monkeypatch):
    """Verifica que el proveedor 'hf_api' instancie HuggingFaceInferenceAPIEmbedding."""
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "dummy_token")
    model = get_embedding_model("hf_api", "BAAI/bge-m3")
    assert isinstance(model, HuggingFaceInferenceAPIEmbedding)
    assert model.model_name == "BAAI/bge-m3"


def test_get_embedding_model_ollama():
    """Verifica que el proveedor 'ollama' instancie OllamaEmbedding."""
    model = get_embedding_model("ollama", "bge-m3")
    assert isinstance(model, OllamaEmbedding)
    assert model.model_name == "bge-m3"


def test_get_embedding_model_openai(monkeypatch):
    """Verifica que el proveedor 'openai' instancie OpenAIEmbedding."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    model = get_embedding_model("openai", "text-embedding-3-small")
    assert isinstance(model, OpenAIEmbedding)
    assert model.model_name == "text-embedding-3-small"


def test_get_embedding_model_invalid():
    """Verifica que un proveedor no soportado lance ValueError con lista de opciones."""
    with pytest.raises(ValueError, match="Proveedor de embedding no soportado"):
        get_embedding_model("unsupported_provider")
