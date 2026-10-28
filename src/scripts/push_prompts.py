"""
src/scripts/push_prompts.py
===========================
Sincroniza y publica todas las plantillas declarativas de prompts (YAML)
desde `src/prompts/` hacia LangSmith Prompt Hub.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
import langsmith
import yaml

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


def push_prompt_file(
    yaml_path: Path,
    client: langsmith.Client | None = None,
    description: str | None = None,
    commit_description: str | None = None,
) -> None:
    """Sube un archivo de prompt YAML específico a LangSmith Hub."""
    client = client or langsmith.Client()
    org = _get_org()
    prefix = f"{org}/" if org else ""

    name, template = load_prompt_from_yaml(yaml_path)
    repo_name = f"{prefix}{name}"

    # Extraer descripción del YAML si no se especificó una por parámetro
    if not description:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            description = data.get("description")

    print(f"Pusheando {repo_name} (desde {yaml_path.name})...")
    url = client.push_prompt(
        repo_name,
        object=template,
        description=description,
        commit_description=commit_description,
    )
    print(f"✅ {repo_name} subido con éxito: {url}")


def push_all_prompts(
    description: str | None = None,
    commit_description: str | None = None,
) -> None:
    """
    Lee todos los archivos *.yaml de src/prompts/ y los sube a LangSmith Hub.
    """
    client = langsmith.Client()

    yaml_files = sorted(list(_PROMPTS_DIR.glob("*.yaml")))
    if not yaml_files:
        print(f"⚠️ No se encontraron plantillas YAML en {_PROMPTS_DIR}")
        return

    print(f"📦 Encontradas {len(yaml_files)} plantillas de prompts para sincronizar.")

    for yaml_path in yaml_files:
        try:
            push_prompt_file(
                yaml_path,
                client=client,
                description=description,
                commit_description=commit_description,
            )
        except Exception as e:
            print(f"❌ Error al subir {yaml_path.name}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pushea plantillas de prompts a LangSmith Hub")
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=None,
        help="Nombre o archivo específico del prompt (ej. ask-clarification)",
    )
    parser.add_argument(
        "--description",
        "-d",
        type=str,
        default=None,
        help="Descripción general del prompt para LangSmith Hub",
    )
    parser.add_argument(
        "--commit-description",
        "-m",
        type=str,
        default=None,
        help="Descripción del commit / versión específica en LangSmith Hub",
    )
    args = parser.parse_args()

    if args.prompt:
        target = args.prompt
        if not target.endswith(".yaml"):
            target_file = _PROMPTS_DIR / f"{target.replace('-', '_')}.yaml"
            if not target_file.exists():
                target_file = _PROMPTS_DIR / f"{target}.yaml"
        else:
            target_file = _PROMPTS_DIR / target

        if not target_file.exists():
            print(f"❌ No se encontró el archivo para el prompt '{args.prompt}' en {_PROMPTS_DIR}")
            return

        push_prompt_file(
            target_file,
            description=args.description,
            commit_description=args.commit_description,
        )
    else:
        push_all_prompts(
            description=args.description,
            commit_description=args.commit_description,
        )


if __name__ == "__main__":
    main()
