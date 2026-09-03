"""
src/scripts/push_prompts.py
===========================
Sincroniza y publica todas las plantillas declarativas de prompts (YAML)
desde `src/prompts/` hacia LangSmith Prompt Hub.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from dotenv import load_dotenv
import langsmith

from src.prompts.loader import load_prompt_from_yaml

logger = logging.getLogger(__name__)

load_dotenv()

_PERSONAL_SENTINEL = {"personal", ""}
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _get_org() -> str | None:
    """
    Lee el handle de organización de LangSmith Hub desde LANGSMITH_HUB_ORG.
    """
    raw = os.getenv("LANGSMITH_HUB_ORG", "").strip()
    if raw.lower() in _PERSONAL_SENTINEL:
        return None  # workspace personal: sin prefijo
    return raw


def push_all_prompts() -> None:
    """
    Lee todos los archivos *.yaml de src/prompts/ y los sube a LangSmith Hub.
    """
    org = _get_org()
    prefix = f"{org}/" if org else ""
    client = langsmith.Client()

    yaml_files = sorted(list(_PROMPTS_DIR.glob("*.yaml")))
    if not yaml_files:
        print(f"⚠️ No se encontraron plantillas YAML en {_PROMPTS_DIR}")
        return

    print(f"📦 Encontradas {len(yaml_files)} plantillas de prompts para sincronizar.")

    for yaml_path in yaml_files:
        try:
            name, template = load_prompt_from_yaml(yaml_path)
            repo_name = f"{prefix}{name}"
            print(f"Pusheando {repo_name} (desde {yaml_path.name})...")
            client.push_prompt(repo_name, object=template)
            print(f"✅ {repo_name} subido con éxito.")
        except Exception as e:
            print(f"❌ Error al subir {yaml_path.name}: {e}")


if __name__ == "__main__":
    push_all_prompts()
