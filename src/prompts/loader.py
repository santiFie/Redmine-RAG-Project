"""
src/prompts/loader.py
=====================
Cargador centralizado de prompts para el proyecto.
Lee las plantillas declarativas en formato YAML ubicadas en `src/prompts/`,
las compila como `ChatPromptTemplate` de LangChain y las almacena en una caché en memoria.
"""

from __future__ import annotations
from langchain_core.prompts.message import BaseMessagePromptTemplate

import asyncio
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent

_ROLE_MAP: dict[str, Any] = {
    "system": SystemMessagePromptTemplate,
    "human": HumanMessagePromptTemplate,
    "user": HumanMessagePromptTemplate,
    "ai": AIMessagePromptTemplate,
    "assistant": AIMessagePromptTemplate,
}


def _create_message_template(role: str, content: str) -> Any:
    role_normalized = role.lower().strip()
    cls = _ROLE_MAP.get(role_normalized)
    if cls is not None:
        return cls.from_template(content)
    return ChatMessagePromptTemplate.from_template(role=role, template=content)


def load_prompt_from_yaml(file_path: Path | str) -> tuple[str, ChatPromptTemplate]:
    """
    Carga un archivo YAML de definición de prompt y lo convierte en ChatPromptTemplate.
    Retorna una tupla con (nombre_prompt, ChatPromptTemplate).
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de prompt: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    prompt_name = data.get("name", path.stem.replace("_", "-"))
    raw_messages = data.get("messages", [])

    message_templates: list[BaseMessagePromptTemplate] = []
    for msg in raw_messages:
        role = msg.get("role", "system")
        content = msg.get("content", "")
        message_templates.append(_create_message_template(role, content))

    chat_prompt = ChatPromptTemplate.from_messages(message_templates)
    return prompt_name, chat_prompt


@lru_cache(maxsize=1)
def load_all_prompts() -> dict[str, ChatPromptTemplate]:
    """
    Escanea el directorio `src/prompts` y carga todas las plantillas YAML disponibles.
    Permite el acceso tanto por el nombre registrado (ej: 'analyze-intent')
    como por el nombre del archivo (ej: 'analyze_intent').
    """
    prompts: dict[str, ChatPromptTemplate] = {}
    for yaml_path in _PROMPTS_DIR.glob("*.yaml"):
        name, template = load_prompt_from_yaml(yaml_path)
        # Registrar con el nombre oficial del prompt (ej: 'triage-duplicate-check')
        prompts[name] = template
        # Registrar también con el nombre de archivo sin extensión y variaciones de guión
        stem = yaml_path.stem
        prompts[stem] = template
        prompts[stem.replace("_", "-")] = template

    return prompts


async def aload_all_prompts() -> dict[str, ChatPromptTemplate]:
    """
    Versión asíncrona de load_all_prompts.
    Delega el escaneo y lectura en disco a un worker thread mediante asyncio.to_thread
    para no bloquear el event loop de asyncio (evitando BlockingError de Blockbuster).
    """
    return await asyncio.to_thread(load_all_prompts)


def _resolve_prompt_from_dict(
    name: str, prompts: dict[str, ChatPromptTemplate]
) -> ChatPromptTemplate:
    # Si viene con prefijo de organización (ej. 'mi-org/analyze-intent'), tomar el nombre base
    base_name = name.split("/")[-1]

    if base_name in prompts:
        return prompts[base_name]

    # Intentar búsqueda con reemplazo de guión/guión bajo
    alt_name = base_name.replace("-", "_")
    if alt_name in prompts:
        return prompts[alt_name]

    alt_dash = base_name.replace("_", "-")
    if alt_dash in prompts:
        return prompts[alt_dash]

    available = sorted(list(set(prompts.keys())))
    raise KeyError(f"Prompt '{name}' no encontrado en {_PROMPTS_DIR}. Disponibles: {available}")


def get_prompt(name: str) -> ChatPromptTemplate:
    """
    Obtiene un ChatPromptTemplate por nombre de forma síncrona.
    Busca primero en las plantillas locales compiladas.
    Lanza KeyError si el prompt solicitado no existe.
    """
    prompts = load_all_prompts()
    return _resolve_prompt_from_dict(name, prompts)


async def aget_prompt(name: str) -> ChatPromptTemplate:
    """
    Obtiene un ChatPromptTemplate por nombre de forma asíncrona.
    Garantiza que la carga y compilación de prompts desde disco no bloquee el event loop.
    Lanza KeyError si el prompt solicitado no existe.
    """
    prompts = await aload_all_prompts()
    return _resolve_prompt_from_dict(name, prompts)
