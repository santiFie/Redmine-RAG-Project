"""
src/redmine/parser.py
=====================
Transformación y refinamiento de respuestas de la API REST de Redmine a nodos de LlamaIndex (TextNodes).

Responsabilidades:
  - Extraer información estructurada (ticket principal, descripciones y comentarios/journals)
  - Mapear atributos de metadatos relevantes para búsqueda y filtrado en RAG
"""

from __future__ import annotations

from typing import Any
from llama_index.core.schema import TextNode


def parse_redmine_issue_to_nodes(issue_data: dict[str, Any]) -> list[TextNode]:
    """
    Toma el JSON de la API REST de Redmine y genera un nodo estructurado
    para el ticket principal y un nodo independiente por cada comentario (journal).

    Args:
        issue_data: Diccionario con la información del issue retornado por Redmine API.

    Returns:
        Lista de TextNodes del dominio listos para ser procesados o indexados.
    """
    nodes: list[TextNode] = []
    issue_id = issue_data["id"]
    project_name = issue_data.get("project", {}).get("name", "")

    # 1. NODO BASE: Título y Descripción del Ticket
    main_text = f"Issue #{issue_id}: {issue_data.get('subject', '')}\n\n{issue_data.get('description', '')}"
    main_node = TextNode(
        text=main_text,
        id_=f"redmine_issue_{issue_id}_main",
        metadata={
            "issue_id": issue_id,
            "entity_type": "issue_description",
            "project": project_name,
            "status": issue_data.get("status", {}).get("name", ""),
            "author": issue_data.get("author", {}).get("name", ""),
        },
    )
    nodes.append(main_node)

    # 2. NODOS POR ENTIDAD: Un nodo independiente por cada comentario/historial (journals)
    for journal in issue_data.get("journals", []):
        notes = journal.get("notes")
        if notes and notes.strip():
            comment_node = TextNode(
                text=f"Comentario en Issue #{issue_id} por {journal.get('user', {}).get('name')}:\n{notes}",
                id_=f"redmine_issue_{issue_id}_journal_{journal['id']}",
                metadata={
                    "issue_id": issue_id,
                    "entity_type": "issue_comment",
                    "project": project_name,
                    "comment_author": journal.get("user", {}).get("name"),
                    "created_on": journal.get("created_on"),
                },
            )
            nodes.append(comment_node)

    return nodes
