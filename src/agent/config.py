"""
src/agent/config.py
===================
Configuración centralizada de proveedores y modelos LLM para los nodos del agente.

Permite gestionar los modelos, proveedores y parámetros (como temperatura)
de cada nodo del flujo, facilitando su personalización mediante variables
de entorno o valores por defecto.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from src.utils.get_llm import get_llm


@dataclass(frozen=True)
class LLMNodeConfig:
    """Configuración de modelo y proveedor para un nodo específico del agente."""

    provider: str
    model: str
    temperature: float = 0.0
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def instantiate(self) -> Any:
        """Instancia el cliente LLM correspondiente usando la factory get_llm."""
        return get_llm(
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            **self.extra_kwargs,
        )


@dataclass
class AgentLLMConfig:
    """Configuración global de todos los modelos LLM utilizados por el agente."""

    # Nodo: Agente ReAct de Redmine (herramientas MCP)
    redmine_agent: LLMNodeConfig = field(
        default_factory=lambda: LLMNodeConfig(
            provider=os.getenv("AGENT_REDMINE_PROVIDER", "gemini"),
            model=os.getenv("AGENT_REDMINE_MODEL", "gemini-2.5-flash"),
            temperature=float(os.getenv("AGENT_REDMINE_TEMP", "0.0")),
        )
    )

    # Nodo: Respuesta basada en documentación recuperada (RAG)
    respond_knowledge: LLMNodeConfig = field(
        default_factory=lambda: LLMNodeConfig(
            provider=os.getenv("AGENT_RAG_RESPOND_PROVIDER", "groq"),
            model=os.getenv("AGENT_RAG_RESPOND_MODEL", "openai/gpt-oss-120b"),
            temperature=float(os.getenv("AGENT_RAG_RESPOND_TEMP", "0.3")),
        )
    )

    # Nodo: Respuesta a intenciones generales (conversación casual)
    respond_general: LLMNodeConfig = field(
        default_factory=lambda: LLMNodeConfig(
            provider=os.getenv("AGENT_GENERAL_RESPOND_PROVIDER", "nvidia"),
            model=os.getenv("AGENT_GENERAL_RESPOND_MODEL", "z-ai/glm-5.3-flash"),
            temperature=float(os.getenv("AGENT_GENERAL_RESPOND_TEMP", "0.5")),
        )
    )

    # Nodo: Evaluación y extracción estructurada de reporte de bug
    qa_evaluator: LLMNodeConfig = field(
        default_factory=lambda: LLMNodeConfig(
            provider=os.getenv("AGENT_QA_PROVIDER", "openrouter"),
            model=os.getenv("AGENT_QA_MODEL", "deepseek/deepseek-v4-flash"),
            temperature=float(os.getenv("AGENT_QA_TEMP", "0.1")),
        )
    )

    # Nodo: Generación de preguntas de clarificación al usuario
    ask_clarification: LLMNodeConfig = field(
        default_factory=lambda: LLMNodeConfig(
            provider=os.getenv("AGENT_CLARIFICATION_PROVIDER", "openrouter"),
            model=os.getenv("AGENT_CLARIFICATION_MODEL", "deepseek/deepseek-v3.2"),
            temperature=float(os.getenv("AGENT_CLARIFICATION_TEMP", "0.2")),
            extra_kwargs={
                "extra_body": {
                    "reasoning": {
                        "effort": os.getenv("AGENT_CLARIFICATION_REASONING", "none"),
                    }
                }
            },
        )
    )

    # Nodo: Detección de tickets duplicados y selección de proyecto
    duplicate_check: LLMNodeConfig = field(
        default_factory=lambda: LLMNodeConfig(
            provider=os.getenv("AGENT_DUPLICATE_CHECK_PROVIDER", "openrouter"),
            model=os.getenv("AGENT_DUPLICATE_CHECK_MODEL", "deepseek/deepseek-v4-flash"),
            temperature=float(os.getenv("AGENT_DUPLICATE_CHECK_TEMP", "0.1")),
        )
    )


# Instancia singleton para uso en el agente
agent_llm_config = AgentLLMConfig()
