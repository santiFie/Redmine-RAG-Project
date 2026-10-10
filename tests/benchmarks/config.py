"""
tests/benchmarks/config.py
==========================
Configuración centralizada de modelos, endpoints y esquemas de preguntas
para el benchmark multimodelo de Decision Models (Issue #12).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from dotenv import load_dotenv
from pydantic import SecretStr

from langchain_typesafe import TypeSafeClassifier
from src.agent.typesafe_config import INTENT_QUESTIONS, SAFE_QUERY_QUESTIONS

load_dotenv()


@dataclass(frozen=True)
class BenchmarkModelConfig:
    """Configuración de conexión y metadatos para un modelo en el benchmark."""

    model_id: str
    display_name: str
    provider: str  # "openrouter", "local", "gcp"
    base_url: str
    api_key: str | None
    timeout: float = 60.0

    def is_configured(self) -> bool:
        """Verifica si el modelo cuenta con los parámetros mínimos para intentar la conexión."""
        if self.provider == "openrouter":
            return bool(self.api_key and self.api_key.strip())
        if self.provider == "gcp":
            return bool(self.base_url and self.base_url.strip())
        # Local asume 127.0.0.1 por defecto
        return bool(self.base_url and self.base_url.strip())

    def create_classifier(self) -> TypeSafeClassifier:
        """Instancia el cliente TypeSafeClassifier correspondiente."""
        resolved_api_key = self.api_key or "dummy-benchmark-key"
        return TypeSafeClassifier(
            model=self.model_id,
            base_url=self.base_url.rstrip("/"),
            api_key=SecretStr(resolved_api_key),
            timeout=self.timeout,
        )


def get_benchmark_models() -> dict[str, BenchmarkModelConfig]:
    """Retorna el registro de los 5 modelos configurados para la comparativa."""
    openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")

    kev_local_base_url = os.getenv("KEV_LOCAL_BASE_URL", "http://127.0.0.1:8009/")
    kev_local_api_key = os.getenv("KEV_LOCAL_API_KEY", "dummy-key")
    kev_local_model = os.getenv("KEV_LOCAL_MODEL", "kev-latest")

    kev_gcp_base_url = os.getenv("KEV_GCP_BASE_URL", "")
    kev_gcp_api_key = os.getenv("KEV_GCP_API_KEY", "dummy-key")
    kev_gcp_model = os.getenv("KEV_GCP_MODEL", "kev-latest")

    return {
        "ms-decision-1": BenchmarkModelConfig(
            model_id="microsoft/microsoft-decision-1",
            display_name="Microsoft Decision 1",
            provider="openrouter",
            base_url=openrouter_base_url,
            api_key=openrouter_api_key,
        ),
        "jev-1.13": BenchmarkModelConfig(
            model_id="typesafe/jev-1.13",
            display_name="TypeSafe JEV 1.13",
            provider="openrouter",
            base_url=openrouter_base_url,
            api_key=openrouter_api_key,
        ),
        "gpt-6-luna": BenchmarkModelConfig(
            model_id="openai/gpt-6-luna-decisions",
            display_name="GPT-6 Luna Decisions",
            provider="openrouter",
            base_url=openrouter_base_url,
            api_key=openrouter_api_key,
        ),
        "kev-0.8b-local": BenchmarkModelConfig(
            model_id=kev_local_model,
            display_name="Kev-0.8B (Local)",
            provider="local",
            base_url=kev_local_base_url,
            api_key=kev_local_api_key,
            timeout=30.0,
        ),
        "kev-4b-gcp": BenchmarkModelConfig(
            model_id=kev_gcp_model,
            display_name="Kev-4B (GCP VM)",
            provider="gcp",
            base_url=kev_gcp_base_url,
            api_key=kev_gcp_api_key,
            timeout=45.0,
        ),
    }


__all__ = [
    "BenchmarkModelConfig",
    "get_benchmark_models",
    "SAFE_QUERY_QUESTIONS",
    "INTENT_QUESTIONS",
]
