"""
src/utils/config.py
===================
Configuración centralizada del proyecto mediante variables de entorno.

Usa Pydantic Settings para validar y tipar todas las variables del .env.
De esta forma, los errores de configuración se detectan al arrancar,
no durante la ejecución.

Uso:
    from src.utils.config import settings

    print(settings.redmine_url)
    print(settings.llm_provider)
"""

from __future__ import annotations

from typing import Literal

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuración global del proyecto.

    Todas las variables se leen automáticamente desde el archivo .env
    o las variables de entorno del sistema.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Redmine
    # ------------------------------------------------------------------
    redmine_url: AnyHttpUrl = Field(
        ..., description="URL base de Redmine (ej. http://localhost:3000)"
    )
    redmine_api_key: str = Field(..., description="API Key de Redmine (Perfil → API access key)")
    redmine_port: int = Field(default=3000)

    # ------------------------------------------------------------------
    # LLM Provider
    # ------------------------------------------------------------------
    llm_provider: Literal["openai", "anthropic", "ollama"] = Field(default="openai")
    llm_model: str = Field(default="gpt-4o-mini")
    openai_api_key: str | None = Field(default=None)
    anthropic_api_key: str | None = Field(default=None)
    ollama_base_url: AnyHttpUrl = Field(default="http://localhost:11434")  # type: ignore
    ollama_model: str = Field(default="llama3.2")

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    embedding_provider: Literal["openai", "ollama"] = Field(default="openai")
    embedding_model: str = Field(default="text-embedding-3-small")

    # ------------------------------------------------------------------
    # Qdrant (Vector DB)
    # ------------------------------------------------------------------
    qdrant_url: AnyHttpUrl = Field(default="http://localhost:6333")  # type: ignore
    qdrant_api_key: str | None = Field(default=None)
    qdrant_collection_name: str = Field(default="redmine_docs")

    # ------------------------------------------------------------------
    # LangSmith / LangChain
    # ------------------------------------------------------------------
    langchain_api_key: str | None = Field(default=None)
    langchain_tracing_v2: bool = Field(default=True)
    langchain_project: str = Field(default="redmine-langgraph-rag")

    # ------------------------------------------------------------------
    # LangGraph Checkpointing
    # ------------------------------------------------------------------
    POSTGRES_URI: str | None = Field(
        default=None,
        description="URI de Postgres para checkpointing de LangGraph",
    )

    # ------------------------------------------------------------------
    # Aplicación
    # ------------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    environment: Literal["development", "production"] = Field(default="development")


# Instancia singleton — importar desde cualquier módulo
# TODO: agregar pydantic-settings a requirements.txt / pyproject.toml
settings = Settings()  # type: ignore[call-arg]  # los valores vienen del .env
